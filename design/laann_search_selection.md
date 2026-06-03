# LAANN Search: Neighbor Selection Philosophy

This document describes the logic by which LAANN selects which pages (megaNodes) to
expand each search iteration, as implemented in
`src/pq_flash_index_laann.cpp` (`laann_search`).

---

## Overview

LAANN operates in two selection modes per iteration — **look-ahead** and **normal** —
and decides which to use via a **persistence check** at the start of each round.
The goal is to extract maximum value from the in-memory page cache before committing
to disk I/O, while falling back to disk whenever a skipped node proves important enough
to affect recall.

---

## The Two Selection Modes

### Look-Ahead Mode

Activated when `--use_look_ahead_search` is set and cached pages exist.

- Scans `retset` forward from the current position.
- Collects **only cached pages** up to `search_beam_width`.
- When an uncached (disk) page is encountered: records its node ID as
  `first_skipped_node_id` but does **not** add it to `pages_to_read` — scanning
  continues in search of further cached pages.
- Result: only `cached_pages` is populated; **no disk I/O is issued this round**.

This allows LAANN to keep making progress through the cached portion of the
graph without paying disk latency on every hop.

### Normal Mode

Standard beam-search selection.

- Scans `retset` forward from the current position.
- Collects **both cached and uncached pages** up to `normal_search_beam_width`.
  - Cached pages → `cached_pages`
  - Uncached pages → `pages_to_read` (async disk reads submitted)
- After filling the beam, does a second scan to locate the next uncached unvisited
  node and records it as `first_skipped_node_id` for the following round.

---

## Persistence Check (Mode Decision Each Round)

At the **start of every iteration**, before selection begins, LAANN decides which
mode to enter:

1. If `first_skipped_node_id` is known and `persistence_window_width > 0`:
   - Scans the top `persistence_window_width` unexpanded nodes in `retset`.
   - If `first_skipped_node_id` is **still present** in the top nodes:
     the skipped node is considered **prominent** → switch to **normal mode**
     so it gets processed with disk I/O.
   - If it has **dropped off** the top:
     it is no longer relevant to the result → remain in **look-ahead mode**.
2. If no `first_skipped_node_id` has been tracked yet → default to look-ahead.

The `persistence_window_width` parameter controls how aggressively LAANN defers disk
I/O: a larger window tolerates skipping uncached nodes for longer before checking
their continued relevance.

### Why the Window Counts All Nodes (Not Just Uncached)

The persistence check scans the top `persistence_window_width` **unvisited** nodes regardless of whether
they are cached or uncached. One might ask: should the window count only uncached nodes, so that
the check is not "diluted" by cached nodes filling up the window?

The answer is no — counting all nodes is the correct semantics. The question the persistence check
is answering is:

> "Is `first_skipped_node_id` still prominent in the **overall** search frontier?"

Using an all-nodes window gives the right answer: if the skipped disk node ranks within the top
`persistence_window_width` candidates globally, it is worth paying disk latency for. If it has been
displaced below that rank by any combination of cached or uncached nodes, it is no longer
a high-priority target.

An uncached-only window would ask a different and weaker question:
"Is it prominent among disk nodes?" — a bar that is easier to clear and would trigger more disk
I/O for nodes that are actually far down the combined ranking. This would reduce the benefit of
look-ahead mode without improving recall.

### Edge Case: No Uncached Unvisited Node Found

In both modes, `new_first_skipped` is initialized to `UINT32_MAX` before the scan
and is only overwritten if an uncached unvisited node is actually encountered.
If the entire remaining frontier consists of cached or already-visited pages,
`first_skipped_node_id` is set to `UINT32_MAX`.

In the next iteration's persistence check, the condition
`first_skipped_node_id != UINT32_MAX` is false, so the else-branch fires and
`look_ahead_mode = true` is forced. This is correct: with no uncached nodes in
the frontier there is nothing on disk to worry about, so look-ahead (or pure
cache processing) is the appropriate behavior. It also means that if the entire
index fits in the cache, disk I/O is never triggered at all.

---

## Neighbor Expansion (After Selection)

Once the set of pages for this round is determined, each page is processed by
`process_pageNode`:

1. **Full-precision vectors** (`page_vectors_buf != nullptr`):
   Compute exact distances for **all** vectors on the page → insert into
   `full_ret_queue` (the final result pool).

2. **Graph neighbors** (`nhood_buf != nullptr`):
   Read the adjacency list from the LAANN disk layout, filter out any neighbor
   whose page is already in `visitedPages`, compute PQ distances for the remaining
   candidates, and insert them into `retset` for future expansion.

Both steps apply to cached and disk pages alike; the difference is only in where
the data comes from (in-memory cache vs. async disk read).

---

## Key Parameters

| Parameter | Role |
|-----------|------|
| `--use_look_ahead_search` | Enables look-ahead mode; without it the search always runs in normal mode |
| `-W` / `search_beam_width` | Unified beam width for all modes (look-ahead and normal) |
| `--persistence_window_width` | How many top candidates to scan when checking if a skipped node is still prominent; `0` = always look-ahead |
| `--num_pages_to_cache` | Number of pages preloaded into the in-memory cache; determines how often look-ahead mode finds cached pages |

---

## Summary

LAANN's selection philosophy is: **stay in cache as long as the skipped disk
nodes are not prominent**. It speculatively defers disk reads by operating purely
on the cache until the persistence check reveals that a bypassed uncached node has
remained near the top of the priority queue long enough to matter. Only then does
it fall back to normal beam search and issue disk I/O. This is the core distinction
from PageANN, which unconditionally issues disk reads every round.
