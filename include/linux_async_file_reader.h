#pragma once
#ifdef __linux__

#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace diskann {

class LinuxAsyncFileReader {
public:
    LinuxAsyncFileReader(const std::string &filename, uint32_t queue_depth = 128);
    ~LinuxAsyncFileReader();

    // Initialize io_uring
    bool init();

    // Prepare async read (adds to SQ, doesn't submit)
    // user_data: pointer to be returned when this I/O completes
    bool prepare_read(char *buf, size_t size, uint64_t offset, void *user_data);

    // Submit all prepared reads in one syscall
    // Returns number of I/Os submitted
    int submit_batch();

    // Poll for completions (non-blocking)
    // completed_user_data: vector to receive user_data pointers for completed I/Os
    // Returns number of completions
    int poll_completions(std::vector<void*> &completed_user_data, const uint64_t max_IOs);

    // Wait for completions (blocking)
    // Blocks until at least one I/O completes
    // completed_user_data: vector to receive user_data pointers for completed I/Os
    // Returns number of completions
    int wait_completions(std::vector<void*> &completed_user_data, const uint64_t max_IOs);

    // Check if initialized
    bool is_initialized() const { return _initialized; }

private:
    io_uring _ring;
    int _fd;
    int _registered_fd_idx;  // Index for registered file descriptor (-1 if not registered)
    uint32_t _queue_depth;
    bool _initialized;
    std::string _filename;
    std::vector<io_uring_cqe*> _cqe_ptrs;  // Pre-allocated CQE pointer array (avoids VLA)
};

} // namespace diskann

#endif // __linux__
