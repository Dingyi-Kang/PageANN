#ifdef __linux__
#include "linux_async_file_reader.h"

namespace diskann {

LinuxAsyncFileReader::LinuxAsyncFileReader(const std::string &filename, uint32_t queue_depth)
    : _queue_depth(queue_depth), _initialized(false), _filename(filename), _registered_fd_idx(-1) {

    // Open file with O_DIRECT for unbuffered I/O (required for async)
    _fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (_fd < 0) {
        throw std::runtime_error("Failed to open file: " + filename + " - " + std::string(strerror(errno)));
    }

    // Pre-allocate CQE pointer array to avoid VLA
    _cqe_ptrs.resize(queue_depth);
}

bool LinuxAsyncFileReader::init() {
    // Setup io_uring in standard mode (no SQPOLL, SQPOLL creates a kernel thread that competes for CPU)
    int ret = io_uring_queue_init(_queue_depth, &_ring, 0);
    if (ret < 0) {
        std::cerr << "io_uring_queue_init failed for " << _filename
                  << ": " << strerror(-ret) << std::endl;
        return false;
    }

    // Register file descriptor for faster I/O (avoids per-request fd lookup in kernel)
    ret = io_uring_register_files(&_ring, &_fd, 1);
    if (ret < 0) {
        // Registration failed - continue without it (fallback to normal fd)
        _registered_fd_idx = -1;
    } else {
        _registered_fd_idx = 0;  // First (and only) registered fd
    }

    _initialized = true;
    return true;
}

bool LinuxAsyncFileReader::prepare_read(char *buf, size_t size, uint64_t offset, void *user_data) {
    if (!_initialized) {
        std::cerr << "ERROR: io_uring not initialized" << std::endl;
        return false;
    }

    // Get a submission queue entry (SQE)
    io_uring_sqe *sqe = io_uring_get_sqe(&_ring);
    if (!sqe) {
        std::cerr << "ERROR: Failed to get SQE (queue full?)" << std::endl;
        return false;
    }

    // Prepare read operation using registered fd if available (faster)
    if (_registered_fd_idx >= 0) {
        io_uring_prep_read(sqe, _registered_fd_idx, buf, size, offset);
        sqe->flags |= IOSQE_FIXED_FILE;  // Tell kernel to use registered fd
    } else {
        io_uring_prep_read(sqe, _fd, buf, size, offset);
    }

    // Attach user_data to identify this I/O when it completes
    io_uring_sqe_set_data(sqe, user_data);

    return true;
}

int LinuxAsyncFileReader::submit_batch() {
    if (!_initialized) {
        std::cerr << "ERROR: io_uring not initialized" << std::endl;
        return -1;
    }

    // Single syscall submits ALL prepared reads!
    int ret = io_uring_submit(&_ring);
    if (ret < 0) {
        std::cerr << "ERROR: io_uring_submit failed: " << strerror(-ret) << std::endl;
        return ret;
    }

    return ret;  // Number of I/Os submitted
}

int LinuxAsyncFileReader::poll_completions(std::vector<void*> &completed_user_data, const uint64_t max_IOs) {
    if (!_initialized) {
        std::cerr << "ERROR: io_uring not initialized" << std::endl;
        return -1;
    }

    // Use pre-allocated array instead of VLA
    unsigned batch_size = std::min((uint64_t)_cqe_ptrs.size(), max_IOs);

    // Non-blocking peek for all available completions (batch mode!)
    unsigned count = io_uring_peek_batch_cqe(&_ring, _cqe_ptrs.data(), batch_size);

    // Process all completions from the batch
    for (unsigned i = 0; i < count; i++) {
        // Check for errors
        if (_cqe_ptrs[i]->res < 0) {
            std::cerr << "ERROR: Async I/O failed: " << strerror(-_cqe_ptrs[i]->res) << std::endl;
        }

        // Get user_data that was set during prepare_read
        void *user_data = io_uring_cqe_get_data(_cqe_ptrs[i]);
        completed_user_data.push_back(user_data);
    }

    // Mark ALL CQEs as seen in one operation (advances CQ head by 'count')
    if (count > 0) {
        io_uring_cq_advance(&_ring, count);
    }

    return count;
}

int LinuxAsyncFileReader::wait_completions(std::vector<void*> &completed_user_data, const uint64_t max_IOs) {
    if (!_initialized) {
        std::cerr << "ERROR: io_uring not initialized" << std::endl;
        return -1;
    }

    // Use pre-allocated array
    unsigned batch_size = std::min((uint64_t)_cqe_ptrs.size(), max_IOs);

    // First, try non-blocking peek to get any already-completed I/Os
    unsigned count = io_uring_peek_batch_cqe(&_ring, _cqe_ptrs.data(), batch_size);

    // If no completions ready, do a blocking wait for at least one
    if (count == 0) {
        io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&_ring, &cqe);
        if (ret < 0) {
            std::cerr << "ERROR: io_uring_wait_cqe failed: " << strerror(-ret) << std::endl;
            return ret;
        }
        // After wait returns, peek again to get all available completions
        count = io_uring_peek_batch_cqe(&_ring, _cqe_ptrs.data(), batch_size);
    }

    // Process all completions
    for (unsigned i = 0; i < count; i++) {
        if (_cqe_ptrs[i]->res < 0) {
            std::cerr << "ERROR: Async I/O failed: " << strerror(-_cqe_ptrs[i]->res) << std::endl;
        }
        void *user_data = io_uring_cqe_get_data(_cqe_ptrs[i]);
        completed_user_data.push_back(user_data);
    }

    // Mark ALL CQEs as seen in one operation
    if (count > 0) {
        io_uring_cq_advance(&_ring, count);
    }

    return count;
}

LinuxAsyncFileReader::~LinuxAsyncFileReader() {
    if (_initialized) {
        // Unregister files before queue exit
        if (_registered_fd_idx >= 0) {
            io_uring_unregister_files(&_ring);
        }
        io_uring_queue_exit(&_ring);
    }
    if (_fd >= 0) {
        close(_fd);
    }
}

} // namespace diskann

#endif // __linux__
