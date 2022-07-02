/*
 * Sort arrays of binary data records.
 *
 * Input and output files contain flat, raw arrays of fixed-length
 * binary data records.
 *
 * Records are interpreted as fixed-length strings of 8-bit unsigned integers.
 * The program sorts these records in lexicographic order.
 *
 * This program is optimized for short records, e.g. up to 20 bytes.
 *
 * Written by Joris van Rantwijk in 2022.
 */


// (already defined by g++)  #define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>


/* Maximum amount of RAM to use (in MBytes). */
#define DEFAULT_MEMORY_SIZE_MBYTE   1024

/* Default branch factor while merging. */
#define DEFAULT_BRANCH_FACTOR       16
#define MAX_BRANCH_FACTOR           16000

/* Default number of sorting threads. */
#define DEFAULT_THREADS             1
#define MAX_THREADS                 128

/* By default use background threads for I/O. */
#define DEFAULT_IO_THREAD           1

/* Align buffer sizes and I/O on this number of records.
   For efficiency, I/O should be done in multiples of 4096 bytes. */
#define TRANSFER_ALIGNMENT          4096

/* Maximum size of I/O buffer during merge passes.
   Big chunks are efficient, but more than 32 MB is probably not helpful. */
#define MAX_MERGE_BUFFER_SIZE       (32 * 1024 * 1024)

/* Template for temporary file name. Must end in 6 'X' characters. */
#define TEMPFILE_TEMPLATE           "sortbin_tmpXXXXXX"


namespace {  // anonymous namespace


/** Information about the sort job. */
struct SortContext
{
    /** Record length in bytes. */
    unsigned int record_size;

    /** Maximum memory to use (bytes). */
    uint64_t memory_size;

    /** Maximum number of arrays to merge in one step. */
    unsigned int branch_factor;

    /** Number of threads for parallel sorting. */
    unsigned int num_threads;

    /** True to eliminate duplicate records. */
    bool flag_unique;

    /** True to do I/O in background threads. */
    bool flag_io_thread;

    /** True to write progress messages to stderr. */
    bool flag_verbose;

    /** Directory where temporary files are created. */
    std::string temporary_directory;
};


/** Strategy for multi-pass external sorting. */
struct SortStrategy
{
    /** Strategy for a single merge pass. */
    struct MergePass {
        /** Number of records in each input block into this pass. */
        std::vector<uint64_t> records_per_block;
    };

    /** Number of records per block during the initial sort pass. */
    uint64_t records_per_sort_block;

    /** Number of blocks for the initial sort pass. */
    uint64_t num_sort_blocks;

    /** Number of blocks that are processed during the first merge pass. */
    uint64_t num_sort_blocks_first_merge;

    /** List of merge passes. */
    std::vector<MergePass> merge_pass;
};


/** Show a progress message. */
void log(const SortContext& ctx, const char *format, ...)
    __attribute__ ((format (printf, 2, 3))) ;

void log(const SortContext& ctx, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    if (ctx.flag_verbose) {
        vfprintf(stderr, format, ap);
    }
    va_end(ap);
}


/** Time measurements. */
class Timer
{
public:
    Timer()
      : m_value(0.0), m_running(false)
    { }

    double value() const
    {
        double v = m_value;
        if (m_running) {
            struct timespec tnow;
            clock_gettime(CLOCK_REALTIME, &tnow);
            v += tnow.tv_sec - m_tstart.tv_sec;
            v += 1.0e-9 * (tnow.tv_nsec - m_tstart.tv_nsec);
        }
        return m_value;
    }

    void start()
    {
        m_value = 0.0;
        m_running = true;
        clock_gettime(CLOCK_REALTIME, &m_tstart);
    }

    void resume()
    {
        if (!m_running) {
            m_running = true;
            clock_gettime(CLOCK_REALTIME, &m_tstart);
        }
    }

    void stop()
    {
        if (m_running) {
            struct timespec tnow;
            clock_gettime(CLOCK_REALTIME, &tnow);
            m_value += tnow.tv_sec - m_tstart.tv_sec;
            m_value += 1.0e-9 * (tnow.tv_nsec - m_tstart.tv_nsec);
            m_running = false;
        }
    }

private:
    double m_value;
    bool m_running;
    struct timespec m_tstart;
};


/** Thread pool for parallel sorting and background I/O. */
class ThreadPool
{
public:
    typedef std::future<void> FutureType;

    /** Initialize thread pool with the specified number of threads. */
    explicit ThreadPool(unsigned int num_threads);

    // Prevent copying and assignment.
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * Destructor: Wait until pending tasks are finished, then join threads.
     */
    ~ThreadPool();

    /**
     * Submit a function to be executed in the thread pool.
     * @param f Function which takes no arguments and returns void.
     * @return A future that becomes ready when the function completes.
     */
    template <class FunctionType>
    FutureType submit(FunctionType f)
    {
        std::packaged_task<void()> task(f);
        FutureType fut(task.get_future());
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push_back(std::move(task));
        m_cond.notify_all();
        return fut;
    }

private:
    /** Worker function that runs in each thread. */
    void worker_function();

    std::mutex                  m_mutex;
    std::condition_variable     m_cond;
    std::vector<std::thread>    m_threads;
    bool                        m_stop_flag;
    std::deque<std::packaged_task<void()>> m_queue;
};

// Constructor.
ThreadPool::ThreadPool(unsigned int num_threads)
  : m_stop_flag(false)
{
    // Create the worker threads.
    for (unsigned int i = 0; i < num_threads; i++) {
        m_threads.emplace_back(&ThreadPool::worker_function, this);
    }
}

// Destructor.
ThreadPool::~ThreadPool()
{
    // Set stop flag.
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop_flag = true;
        m_cond.notify_all();
    }

    // Join the worker threads.
    for (auto& thread : m_threads) {
        thread.join();
    }
}

// Worker function that runs in each thread.
void ThreadPool::worker_function()
{
    // Main loop.
    while (true) {

        std::packaged_task<void()> task;

        // Get a pending task from the queue.
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            // Wait until the queue is non-empty or the stop flag is raised.
            while (m_queue.empty() && !m_stop_flag) {
                m_cond.wait(lock);
            }

            // Exit if the stop flag is raised (thread pool is shutting down).
            if (m_stop_flag) {
                break;
            }

            // Get oldest pending task.
            task = std::move(m_queue.front());
            m_queue.pop_front();
        }

        // Execute the pending task.
        task();
    }
}


/** Helper class to wait until multiple "std::future"s are ready. */
class CompletionTracker
{
public:
    typedef std::future<void> FutureType;

    /** Add a future to wait for. */
    void add(FutureType&& fut)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_futures.emplace_back(std::move(fut));
    }

    /** Wait until all futures have completed. */
    void wait()
    {
        while (true) {
            FutureType fut;

            // Get the next future from the queue.
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                if (m_futures.empty()) {
                    // No more futures. We are done waiting.
                    break;
                }

                fut = std::move(m_futures.front());
                m_futures.pop_front();
            }

            // Wait on this future.
            fut.get();
        }
    }

private:
    std::mutex m_mutex;
    std::deque<FutureType> m_futures;
};


/**
 * Binary file access.
 *
 * This is a base class which can not be directly constructed.
 * Subclasses implement constructors that open files in various ways.
 */
class BinaryFile
{
public:
    // Prevent copying and assignment.
    BinaryFile(const BinaryFile&) = delete;
    BinaryFile& operator=(const BinaryFile&) = delete;

    /** Return file size in bytes. */
    uint64_t size() const
    {
        return m_file_size;
    }

    /** Truncate file to reduce its size. */
    void truncate_file(uint64_t new_size);

    /** Read raw bytes from file. */
    void read(
        unsigned char * buf,
        uint64_t file_offset,
        size_t length);

    /** Write raw bytes to file. */
    void write(
        const unsigned char * buf,
        uint64_t file_offset,
        size_t length);

protected:
    // Prevent constructing and destructing via this base class.
    BinaryFile()
        : m_fd(-1), m_file_size(0)
    { }

    /** Destructor: close file. */
    ~BinaryFile()
    {
        if (m_fd >= 0) {
            close(m_fd);
        }
    }

    /** File descriptor. */
    int         m_fd;

    /** File size (bytes). */
    uint64_t    m_file_size;
};


// Truncate file to reduce its size.
void BinaryFile::truncate_file(uint64_t new_size)
{
    int ret = ftruncate(m_fd, new_size);
    if (ret != 0) {
        throw std::system_error(
            errno,
            std::system_category(),
            "Can not truncate file");
    }
    m_file_size = new_size;
}

// Read raw bytes from file.
void BinaryFile::read(
    unsigned char * buf,
    uint64_t file_offset,
    size_t length)
{
    // Note that pread() may read fewer bytes than requested.
    while (length > 0) {
        ssize_t ret = pread(m_fd, buf, length, file_offset);
        if (ret <= 0) {
            throw std::system_error(
                errno,
                std::system_category(),
                "Can not read from file");
        }
        buf += ret;
        file_offset += ret;
        length -= ret;
    }
}

// Write raw bytes to file.
void BinaryFile::write(
    const unsigned char * buf,
    uint64_t file_offset,
    size_t length)
{
    // Note that pwrite() may write fewer bytes than requested.
    while (length > 0) {
        ssize_t ret = pwrite(m_fd, buf, length, file_offset);
        if (ret <= 0) {
            throw std::system_error(
                errno,
                std::system_category(),
                "Can not write to file");
        }
        buf += ret;
        file_offset += ret;
        length -= ret;
    }
}


/** Binary input file. */
class BinaryInputFile : public BinaryFile
{
public:
    /** Open existing file for read-only access. */
    explicit BinaryInputFile(const std::string& filename)
    {
        m_fd = open(filename.c_str(), O_RDONLY);
        if (m_fd < 0) {
            throw std::system_error(
                errno,
                std::system_category(),
                "Can not open input file");
        }

        off_t fsz = lseek(m_fd, 0, SEEK_END);
        if (fsz == (off_t)(-1)) {
            throw std::system_error(
                errno,
                std::system_category(),
                "Can not seek input file");
        }

        m_file_size = fsz;
    }
};


/** Binary output file. */
class BinaryOutputFile : public BinaryFile
{
public:
    /** Create output file and pre-allocate space. */
    BinaryOutputFile(const std::string& filename, uint64_t new_size)
    {
        m_fd = open(filename.c_str(), O_RDWR | O_CREAT | O_EXCL, 0666);
        if (m_fd < 0) {
            throw std::system_error(
                errno,
                std::system_category(),
                "Can not create output file");
        }

        int ret = posix_fallocate(m_fd, 0, new_size);
        if (ret != 0) {
            int errnum = errno;

            // Delete empty output file.
            unlink(filename.c_str());

            throw std::system_error(
                errnum,
                std::system_category(),
                "Can not allocate space in output file");
        }

        m_file_size = new_size;
    }
};


/** Temporary binary file. */
class BinaryTempFile : public BinaryFile
{
public:
    /** Create temporary file and pre-allocate space. */
    BinaryTempFile(const std::string& tempdir, uint64_t new_size)
    {
        // Prepare file name template ending in 6 'X' characters.
        std::string filename_template = tempdir;
        if (!filename_template.empty() && filename_template.back() != '/') {
            filename_template.push_back('/');
        }
        filename_template.append(TEMPFILE_TEMPLATE);

        // Copy template to modifiable buffer.
        std::vector<char> filename_buf(
            filename_template.c_str(),
            filename_template.c_str() + filename_template.size() + 1);

        // Create temporary file with unique name.
        // mkstemp() replaces the 'X' characters in the file name template
        // to create a unique file name.
        m_fd = mkstemp(filename_buf.data());
        if (m_fd < 0) {
            throw std::system_error(
                errno,
                std::system_category(),
                "Can not create temporary file");
        }

        int ret = posix_fallocate(m_fd, 0, new_size);
        if (ret != 0) {
            int errnum = errno;

            // Delete temporary file.
            unlink(filename_buf.data());

            throw std::system_error(
                errnum,
                std::system_category(),
                "Can not allocate space in temporary file");
        }

        m_file_size = new_size;

        // Delete the temporary file.
        // Since the file is still open, it will continue to exist on disk
        // until the file handle is closed.
        if (unlink(filename_buf.data()) != 0) {
            throw std::system_error(
                errno,
                std::system_category(),
                "Can not delete temporary file");
        }
    }
};


/**
 * Read binary records from an input file with buffering.
 *
 * The input stream reads from a sequence of discontinuous blocks in
 * the input file. Each block contains a flat array of binary records.
 * An explicit list of these blocks is passed to the stream constructor.
 *
 * The input stream starts in the "empty" state.
 * The first call to "next_block()" enables reading records from the
 * first input block. When the input stream reaches the end of the
 * current block, it again enters the "empty" state until the following
 * call to "next_block()".
 */
class RecordInputStream
{
public:
    typedef std::vector<std::tuple<uint64_t, uint64_t>> BlockList;

    /**
     * Construct a record input stream.
     *
     * The stream will initially contain no input sections.
     *
     * @param input_file    Input file where records read from.
     * @param record_size   Record size in bytes.
     * @param blocks        Vector of input blocks specified as tuple
     *                      (file_offset, number_of_records).
     * @param buffer_size   Buffer size in bytes.
     *                      Must be a multiple of "record_size".
     *                      Note: If an "io_pool" is available, each input
     *                      stream allocates 2 buffers of the specified size,
     *                      otherwise just 1 buffer.
     * @param io_pool       Optional thread pool for background I/O.
     */
    RecordInputStream(
        BinaryFile&   input_file,
        unsigned int  record_size,
        BlockList&&   blocks,
        size_t        buffer_size,
        ThreadPool *  io_pool)
      : m_input_file(input_file),
        m_record_size(record_size),
        m_next_block(0),
        m_block_remaining(0),
        m_file_offset(0),
        m_bufpos(NULL),
        m_bufend(NULL),
        m_blocks(blocks),
        m_buffer_size(buffer_size),
        m_buffer_index(0),
        m_io_pool(io_pool)
    {
        assert(buffer_size % record_size == 0);
        assert(buffer_size > record_size);

        // Allocate buffer(s).
        m_buffers[0].resize(buffer_size);
        if (io_pool != NULL) {
            // Background I/O: allocate extra buffer.
            m_buffers[1].resize(buffer_size);

            // Start reading in background thread.
            pre_read_buffer();
        }
    }

    // Prevent copying and assignment.
    RecordInputStream(const RecordInputStream&) = delete;
    RecordInputStream& operator=(const RecordInputStream&) = delete;

    /** Return true if the end of the current input block is reached. */
    inline bool empty() const
    {
        return (m_bufpos == m_bufend);
    }

    /**
     * Return a pointer to the current binary record.
     *
     * This function must only be used if "end_of_stream()" returns false.
     * The returned pointer becomes invalid after a call to "next_record()".
     */
    inline const unsigned char * record() const
    {
        return m_bufpos;
    }

    /**
     * Move to the next record of the current section.
     *
     * This function must only be used if "empty()" returns false.
     * Calling this function invalidates all pointers previously returned
     * by "current_record()".
     */
    inline void next_record()
    {
        assert(m_bufpos != m_bufend);
        m_bufpos += m_record_size;

        if (m_bufpos == m_bufend) {
            refill_buffer();
        }
    }

    /**
     * Start reading from the next input section.
     *
     * This function may only be called if "empty()" returns true.
     */
    void next_block()
    {
        assert(m_bufpos == m_bufend);
        assert(m_next_block < m_blocks.size());

        uint64_t num_records;
        std::tie(m_file_offset, num_records) = m_blocks[m_next_block];
        m_block_remaining = num_records * m_record_size;
        m_next_block++;

        uint64_t file_size = m_input_file.size();
        assert(m_file_offset <= file_size);
        assert(m_block_remaining <= file_size - m_file_offset);

        refill_buffer();
    }

private:
    /** Refill the buffer from the current input section. */
    void refill_buffer()
    {
        if (m_block_remaining > 0) {
            size_t transfer_size = std::min(uint64_t(m_buffer_size),
                                            m_block_remaining);

            if (m_io_pool == NULL) {
                // Use blocking I/O.
                m_input_file.read(
                    m_buffers[0].data(),
                    m_file_offset,
                    transfer_size);
            } else {
                // The data we need is already on its way.
                // Wait until the background transfer completes.
                assert(m_future.valid());
                m_future.get();

                // Flip to the refilled buffer.
                m_buffer_index = (m_buffer_index + 1) & 1;
            }

            // Update position administration.
            m_file_offset += transfer_size;
            m_block_remaining -= transfer_size;

            // Update buffer pointers.
            m_bufpos = m_buffers[m_buffer_index].data();
            m_bufend = m_bufpos + transfer_size;

            // Start reading next buffer in background thread.
            if (m_io_pool != NULL) {
                pre_read_buffer();
            }
        }
    }

    /** Start reading into the next buffer. */
    void pre_read_buffer()
    {
        assert(m_io_pool != NULL);

        uint64_t pre_file_offset = m_file_offset;
        uint64_t pre_block_remaining = m_block_remaining;
        size_t   pre_block_index = m_next_block;

        // If we are at the end of the current block,
        // look ahead to the next block.
        while (pre_block_remaining == 0) {

            if (pre_block_index >= m_blocks.size()) {
                // Reached the end of the last block.
                // There is nothing more to read in this stream.
                return;
            }

            // Look ahead to the next block.
            uint64_t num_records;
            std::tie(pre_file_offset, num_records) = m_blocks[pre_block_index];
            pre_block_remaining = num_records * m_record_size;
            pre_block_index++;

            uint64_t file_size = m_input_file.size();
            assert(pre_file_offset <= file_size);
            assert(pre_block_remaining <= file_size - pre_file_offset);
        }

        // Read into the currently non-active buffer.
        unsigned int pre_buffer_index = (m_buffer_index + 1) & 1;

        size_t transfer_size = std::min(uint64_t(m_buffer_size),
                                        pre_block_remaining);

        // Start background I/O.
        m_future = m_io_pool->submit(
            std::bind(
                &BinaryFile::read,
                &m_input_file,
                m_buffers[pre_buffer_index].data(),
                pre_file_offset,
                transfer_size));
    }

    BinaryFile&         m_input_file;
    const unsigned int  m_record_size;
    size_t              m_next_block;
    uint64_t            m_block_remaining;
    uint64_t            m_file_offset;
    unsigned char *     m_bufpos;
    unsigned char *     m_bufend;
    BlockList           m_blocks;
    size_t              m_buffer_size;
    unsigned int        m_buffer_index;
    std::array<std::vector<unsigned char>, 2> m_buffers;
    ThreadPool::FutureType m_future;
    ThreadPool *        m_io_pool;
};


/**
 * Write binary records to an output file with buffering.
 */
class RecordOutputStream
{
public:
    /**
     * Construct a record output stream.
     *
     * @param output_file   Output file where records are written.
     * @param record_size   Record size in bytes.
     * @param file_offset   Start offset in the output file.
     * @param buffer_size   Buffer size in bytes.
     * @param num_buffers   Number of buffers of the specified size.
     *                      Note: Without "io_pool", the output stream
     *                      uses only 1 buffer.
     * @param io_pool       Optional thread pool for background I/O.
     */
    RecordOutputStream(
        BinaryFile&   output_file,
        unsigned int  record_size,
        uint64_t      file_offset,
        size_t        buffer_size,
        unsigned int  num_buffers,
        ThreadPool *  io_pool)
      : m_output_file(output_file),
        m_record_size(record_size),
        m_file_offset(file_offset),
        m_buffer_index(0),
        m_io_pool(io_pool)
    {
        assert(buffer_size > record_size);
        assert(num_buffers > 0);

        if (io_pool == NULL) {
            // No background I/O: Use just 1 buffer.
            m_buffers.emplace_back(buffer_size);
        } else {
            // Allocate buffers and futures.
            for (unsigned int b = 0; b < num_buffers; b++) {
                m_buffers.emplace_back(buffer_size);
                m_futures.emplace_back();
            }
        }

        m_bufpos = m_buffers[0].data();
        m_bufend = m_bufpos + m_buffers[0].size();
    }

    // Prevent copying and assignment.
    RecordOutputStream(const RecordOutputStream&) = delete;
    RecordOutputStream& operator=(const RecordOutputStream&) = delete;

    /** Append a record to the output stream. */
    inline void put(const unsigned char *record)
    {
        if (m_record_size > m_bufend - m_bufpos) {
            flush_block();
        }
        memcpy(m_bufpos, record, m_record_size);
        m_bufpos += m_record_size;
    }

    /** Return the current file offset. Flush before calling this function. */
    inline uint64_t file_offset() const
    {
        // File offset is only accurate when the current buffer is empty.
        assert(m_bufpos == m_buffers[m_buffer_index].data());
        return m_file_offset;
    }

    /** Flush buffered records to the output file. */
    void flush()
    {
        // Write the current block to file.
        flush_block();

        // Wait until all background I/O complete.
        if (m_io_pool != NULL) {
            for (ThreadPool::FutureType& fut : m_futures) {
                if (fut.valid()) {
                    fut.get();
                }
            }
        }
    }

private:
    /** Write the current block to file and obtain an empty buffer. */
    void flush_block()
    {
        unsigned char * buf_start = m_buffers[m_buffer_index].data();
        size_t flush_size = m_bufpos - buf_start;

        if (flush_size > 0) {

            // Write the current block to file.
            if (m_io_pool == NULL) {
                // Use blocking I/O.
                m_output_file.write(buf_start, m_file_offset, flush_size);
            } else {
                // Use background I/O.
                m_futures[m_buffer_index] = m_io_pool->submit(
                    std::bind(
                        &BinaryFile::write,
                        &m_output_file,
                        buf_start,
                        m_file_offset,
                        flush_size));

                // Move on to the next buffer.
                m_buffer_index = (m_buffer_index + 1) % m_buffers.size();

                // Wait until the new buffer is available.
                if (m_futures[m_buffer_index].valid()) {
                    m_futures[m_buffer_index].get();
                }
            }

            // Update file offset.
            m_file_offset += flush_size;

            // Update buffer pointers.
            m_bufpos = m_buffers[m_buffer_index].data();
            m_bufend = m_bufpos + m_buffers[m_buffer_index].size();
        }
    }

    BinaryFile&         m_output_file;
    const unsigned int  m_record_size;
    uint64_t            m_file_offset;
    unsigned char *     m_bufpos;
    unsigned char *     m_bufend;
    unsigned int        m_buffer_index;
    std::vector<std::vector<unsigned char>> m_buffers;
    std::vector<ThreadPool::FutureType> m_futures;
    ThreadPool *        m_io_pool;
};


/** Compare two records. */
#define record_compare(_a, _b, _n)  (memcmp(_a, _b, _n))

/** Copy a record. */
#define record_copy(_dst, _src, _n) (memcpy(_dst, _src, _n))


/**
 * Swap two records.
 */
inline void record_swap(
    unsigned char *a,
    unsigned char *b,
    size_t record_size)
{
    while (record_size > 0) {
        unsigned char aa = *a;
        unsigned char bb = *b;
        *b = aa;
        *a = bb;
        a++;
        b++;
        record_size--;
    }
}


/** Return the index of the parent of the specified heap node. */
inline size_t heap_parent_index(size_t node_index)
{
    return (node_index - 1) / 2;
}


/** Return the index of the left child of the specified heap node. */
inline size_t heap_left_child_index(size_t node_index)
{
    return node_index * 2 + 1;
}


/**
 * Insert a node into an empty spot in the heap, then repair the sub-heap
 * rooted at that position.
 */
void heap_sift_down_records(
    unsigned char * buffer,
    size_t record_size,
    size_t num_records,
    size_t insert_index,
    const unsigned char * insert_value)
{
    // Find the first node index which does not have two child nodes.
    size_t parent_end = heap_parent_index(num_records);

    // Move the empty spot all the way down through the sub-heap.
    size_t cur_idx = insert_index;
    while (cur_idx < parent_end) {

        // Find the left child node of the current node.
        size_t child_idx = heap_left_child_index(cur_idx);
        unsigned char * child_ptr = buffer + child_idx * record_size;

        // Compare the two child nodes.
        if (record_compare(child_ptr, child_ptr + record_size, record_size)
            < 0) {
            // Right child is greater, choose that one.
            child_idx += 1;
            child_ptr += record_size;
        }

        // Move the chosen child to the empty spot.
        unsigned char * cur_ptr = buffer + cur_idx * record_size;
        record_copy(cur_ptr, child_ptr, record_size);

        // Continue the scan from the (now empty) child spot.
        cur_idx = child_idx;
    }

    // If the empty spot has a left child, swap it with the empty spot.
    if (num_records > 1 && cur_idx <= heap_parent_index(num_records - 1)) {
        size_t child_idx = heap_left_child_index(cur_idx);
        unsigned char * cur_ptr = buffer + cur_idx * record_size;
        unsigned char * child_ptr = buffer + child_idx * record_size;
        record_copy(cur_ptr, child_ptr, record_size);
        cur_idx = child_idx;
    }

    // The empty spot is now in a leaf node of the heap.
    // Scan back up to find the right place to insert the new node.
    //
    // Going all the way down and then back up may seem wasteful,
    // but it is faster in practice because the correct insertion spot
    // is likely to be in the bottom of the heap.

    while (cur_idx > insert_index) {

        // Find parent of the empty spot.
        size_t parent_idx = heap_parent_index(cur_idx);
        unsigned char * parent_ptr = buffer + parent_idx * record_size;

        // Compare the new value to the parent value.
        if (record_compare(parent_ptr, insert_value, record_size) >= 0) {
            // We found the right spot to insert the new value.
            break;
        }

        // Move the parent back to the empty spot.
        unsigned char * cur_ptr = buffer + cur_idx * record_size;
        record_copy(cur_ptr, parent_ptr, record_size);

        // Move to te parent node.
        cur_idx = parent_idx;
    }

    // Insert the new node at the empty position.
    unsigned char * cur_ptr = buffer + cur_idx * record_size;
    record_copy(cur_ptr, insert_value, record_size);
}


/**
 * Sort an array of records using in-place heap sort.
 *
 * Run time:    O(N * log(N))
 * Extra space: O(1)
 */
void heap_sort_records(
    unsigned char * buffer,
    size_t record_size,
    size_t num_records)
{
    // Skip trivial cases.
    if (num_records < 2) {
        return;
    }

    // Allocate temporary space for one record.
    std::vector<unsigned char> temp(record_size);

    //
    // Phase 1: Transform the unordered array into a max-heap.
    //

    // Bottom-up loop over all non-trivial sub-heaps.
    // Start with the parent of the last node.
    size_t cur_idx = heap_parent_index(num_records - 1);

    while (true) {

        // Remove the current node from the heap.
        unsigned char * cur_ptr = buffer + cur_idx * record_size;
        record_copy(temp.data(), cur_ptr, record_size);

        // Re-insert the node and repair the sub-heap rooted at this node.
        heap_sift_down_records(
            buffer,
            record_size,
            num_records,
            cur_idx,
            temp.data());

        // Stop after processing the root node.
        if (cur_idx == 0) {
            break;
        }

        // Go do the next sub-heap.
        cur_idx--;
    }

    //
    // Phase 2: Transform the max-heap into a sorted array.
    //

    // Loop over the record array from back to front.
    cur_idx = num_records - 1;
    while (cur_idx > 0) {

        // Remove the root node from the heap.
        // This is the largest remaining element, which belongs at index CUR
        // in the sorted array.
        record_copy(temp.data(), buffer, record_size);

        // Remove the node at index CUR from the heap.
        // This reduces the size of the heap by 1.
        // Re-insert the removed node as the root node and repair the heap.
        unsigned char * cur_ptr = buffer + cur_idx * record_size;
        heap_sift_down_records(buffer, record_size, cur_idx, 0, cur_ptr);

        // Copy the former root node to its position in the sorted array.
        record_copy(cur_ptr, temp.data(), record_size);

        // Go to next element.
        cur_idx--;
    }
}


/**
 * Sort an array of records using in-place insertion sort.
 *
 * This is a helper function for quicksort_records().
 */
void insertion_sort_records(
    unsigned char * buffer,
    size_t record_size,
    size_t num_records)
{
    // Allocate temporary space for one record.
    std::vector<unsigned char> temp(record_size);

    for (size_t cur_idx = 1; cur_idx < num_records; cur_idx++) {
        // The partial array 0 .. (cur_idx - 1) is already sorted.
        // We will now insert cur_idx into the sorted array.

        // Quick check whether the new record is already in the right place.
        unsigned char * cur_ptr = buffer + cur_idx * record_size;
        unsigned char * prev_ptr = cur_ptr - record_size;
        if (record_compare(cur_ptr, prev_ptr, record_size) >= 0) {
            continue;
        }

        // Scan backwards through the sorted array to find the right place
        // to insert the new record.
        unsigned char * insert_ptr = prev_ptr;
        while (insert_ptr > buffer) {

            prev_ptr = insert_ptr - record_size;
            if (record_compare(cur_ptr, prev_ptr, record_size) >= 0) {
                // Found the right place to insert the new record.
                break;
            }

            // Continue backwards scan.
            insert_ptr = prev_ptr;
        }

        // Copy the new record to temporary storage.
        record_copy(temp.data(), cur_ptr, record_size);

        // Move sorted records to make space for the new record.
        memmove(insert_ptr + record_size, insert_ptr, cur_ptr - insert_ptr);

        // Copy the new record to its place.
        record_copy(insert_ptr, temp.data(), record_size);
    }
}


/**
 * Partition the array into two parts.
 *
 * This is a helper function for quicksort_records().
 *
 * After partitioning, all of the "num_left" first records will be
 * less-than-or-equal to all of the "num_right" last records.
 *
 * After partitioning, either
 *   num_left + num_right == num_records
 * or
 *   num_left + num_right == num_records - 1
 *
 * In the second case, the record in the middle is already in its final
 * position in the array.
 *
 * @param range_start       Pointer to first element of the array.
 * @param record_size       Record size in bytes.
 * @param num_records       Number of records in the array.
 * @param[out] num_left     Number of records in the left half.
 * @param[out] num_right    Number of records in the right half.
 */
inline void quicksort_partition_records(
    unsigned char * range_begin,
    size_t record_size,
    size_t num_records,
    size_t& num_left,
    size_t& num_right)
{
    // Initialize pointers to start, end and middle of range.
    unsigned char * left_ptr = range_begin;
    unsigned char * right_ptr = range_begin + (num_records - 1) * record_size;
    unsigned char * pivot_ptr = range_begin + (num_records / 2) * record_size;

    // Sort the first, middle and last records such that they are
    // in proper order with respect to each other.
    if (record_compare(pivot_ptr, left_ptr, record_size) < 0) {
        record_swap(left_ptr, pivot_ptr, record_size);
    }
    if (record_compare(right_ptr, pivot_ptr, record_size) < 0) {
        record_swap(pivot_ptr, right_ptr, record_size);
        if (record_compare(pivot_ptr, left_ptr, record_size) < 0) {
            record_swap(left_ptr, pivot_ptr, record_size);
        }
    }

    // The median of the three records we examined is now in the
    // middle of the range, pointed to by pivot_ptr.
    // This is not necessarily the final location of that element.

    // The first and last record of the range are now on the proper
    // side of the partition. No need to examine them again.
    left_ptr += record_size;
    right_ptr -= record_size;

    // Partition the rest of the array based on comparing to the pivot.
    while (true) {

        // Skip left-side records that are less than the pivot.
        while (record_compare(left_ptr, pivot_ptr, record_size) < 0) {
            left_ptr += record_size;
        }

        // Skip right-side records that are greater than the pivot.
        while (record_compare(pivot_ptr, right_ptr, record_size) < 0) {
            right_ptr -= record_size;
        }

        // Stop when the pointers meet.
        if (left_ptr >= right_ptr) {
            break;
        }

        // Swap the records that are on the wrong sides.
        record_swap(left_ptr, right_ptr, record_size);

        // If we moved the pivot, update its pointer so it keeps
        // pointing to the pivot value.
        if (pivot_ptr == left_ptr) {
            pivot_ptr = right_ptr;
        } else if (pivot_ptr == right_ptr) {
            pivot_ptr = left_ptr;
        }

        // Do not compare the swapped elements again.
        left_ptr += record_size;
        right_ptr -= record_size;

        // Stop when pointers cross.
        // (Pointers equal is not good enough at this point, because
        //  we won't know on which side the pointed record belongs.)
        if (left_ptr > right_ptr) {
            break;
        }
    }

    // If pointers are equal, they must both be pointing to a pivot.
    // Bump both pointers so they correctly delineate the new
    // subranges. The record where the pointers meet is already in
    // its final position.
    if (left_ptr == right_ptr) {
        left_ptr += record_size;
        right_ptr -= record_size;
    }

    // Determine the number of elements in the left and right subranges.
    num_left = (right_ptr + record_size - range_begin) / record_size;
    num_right = num_records - (left_ptr - range_begin) / record_size;
}


/**
 * Sort an array of records using in-place quicksort.
 *
 * Run time:    O(N * log(N))
 * Extra space: O(log(N))
 *
 * Plain quicksort is known to have quadratic worst-case behaviour.
 * This implementation uses median-of-three partitioning to reduce
 * the probability of worst-case performance. If bad performance does
 * occur, this implementation detects it and switches to heap sort.
 */
void quicksort_records(
    unsigned char * buffer,
    size_t record_size,
    size_t num_records)
{
    // Recursive partitioning is only applied to fragments larger than
    // this threshold. The rest of the work will be done by insertion sort.
    const size_t insertion_sort_threshold = 12;

    // Skip trivial cases.
    if (num_records < 2) {
        return;
    }

    // Determine maximum acceptable recursion depth.
    unsigned int depth_limit = 1;
    for (size_t nremain = num_records; nremain > 1; nremain >>= 1) {
        depth_limit += 2;
    }

    // Allocate stack.
    std::vector<std::tuple<unsigned char *, size_t, unsigned int>> stack;
    stack.reserve(depth_limit);

    // Prepare recursive partitioning of the entire array.
    if (num_records > insertion_sort_threshold) {
        stack.emplace_back(buffer, num_records, depth_limit);
    }

    // Execute recursive partitioning.
    while (!stack.empty()) {

        // Pop a range from the stack.
        unsigned char * range_begin;
        size_t range_num_records;
        std::tie(range_begin, range_num_records, depth_limit) = stack.back();
        stack.pop_back();

        // Check recursion depth. Switch to heap sort if we get to deep.
        if (depth_limit == 0) {
            heap_sort_records(range_begin, record_size, range_num_records);
            continue;
        }

        // Partition the array into two parts.
        size_t num_left, num_right;
        quicksort_partition_records(
            range_begin,
            record_size,
            range_num_records,
            num_left,
            num_right);

        // Push left subrange on the stack, if it meets the size threshold.
        if (num_left > insertion_sort_threshold) {
            stack.emplace_back(range_begin, num_left, depth_limit - 1);
        }

        // Push right subrange on the stack, if it meets the size threshold.
        if (num_right > insertion_sort_threshold) {
            unsigned char * right_half =
                range_begin + (range_num_records - num_right) * record_size;
            stack.emplace_back(right_half, num_right, depth_limit - 1);
        }
    }

    // Recursive partitining finished.
    // The array is now roughly sorted, except for subranges that were
    // skipped because they are within the threshold.

    // Finish with insertion sort to get a fully sorted array.
    insertion_sort_records(buffer, record_size, num_records);
}


/** Helper function for quicksort_records_parallel(). */
void quicksort_records_parallel_step(
    unsigned char * range_begin,
    size_t record_size,
    size_t num_records,
    size_t parallel_size_threshold,
    unsigned int parallel_depth_limit,
    ThreadPool * thread_pool,
    CompletionTracker * completion_tracker)
{
    while (true) {

        // If the range is below threshold, or recursion is too deep,
        // handle this part of the array entirely in this thread.
        if (num_records <= parallel_size_threshold
                || parallel_depth_limit == 0) {
            quicksort_records(range_begin, record_size, num_records);
            break;
        }

        parallel_depth_limit--;

        // Partition the array into two parts.
        size_t num_left, num_right;
        quicksort_partition_records(
            range_begin,
            record_size,
            num_records,
            num_left,
            num_right);

        // Submit the largest of the two subranges to the thread pool.
        // We will handle the other subrange within this thread.
        unsigned char * right_half =
            range_begin + (num_records - num_right) * record_size;
        if (num_left >= num_right) {

            // Submit left subrange.
            completion_tracker->add(
                thread_pool->submit(
                    std::bind(
                        quicksort_records_parallel_step,
                        range_begin,
                        record_size,
                        num_left,
                        parallel_size_threshold,
                        parallel_depth_limit,
                        thread_pool,
                        completion_tracker)));

            // Continue with right subrange in this thread.
            range_begin = right_half;
            num_records = num_right;

        } else {

            // Submit right subrange.
            completion_tracker->add(
                thread_pool->submit(
                    std::bind(
                        quicksort_records_parallel_step,
                        right_half,
                        record_size,
                        num_right,
                        parallel_size_threshold,
                        parallel_depth_limit,
                        thread_pool,
                        completion_tracker)));

            // Continue with left subrange in this thread.
            num_records = num_left;
        }
    }
}


/**
 * Sort an array of records using in-place quicksort.
 *
 * Use multiple threads to parallelize the sort process.
 */
void quicksort_records_parallel(
    unsigned char * buffer,
    size_t record_size,
    size_t num_records,
    unsigned int num_threads,
    ThreadPool * thread_pool)
{
    // Small fragments should not be further distributed between threads.
    size_t parallel_size_threshold =
        std::max(size_t(1024), num_records / num_threads / 4);

    // Stop parallel processing past a certain nesting depth.
    // This is necessary to avoid quadratic run time.
    unsigned int parallel_depth_limit = 2;
    for (unsigned int nn = num_threads; nn > 1; nn >>= 1) {
        parallel_depth_limit += 2;
    }

    // Tracker to determine when all sort tasks have finished.
    CompletionTracker completion_tracker;

    // Submit the full array to the thread pool.
    completion_tracker.add(
        thread_pool->submit(
            std::bind(
                quicksort_records_parallel_step,
                buffer,
                record_size,
                num_records,
                parallel_size_threshold,
                parallel_depth_limit,
                thread_pool,
                &completion_tracker)));

    // Wait until all sort tasks have finished.
    completion_tracker.wait();
}


/**
 * Sort the specified block of records (in-place).
 *
 * Optionally use a thread pool for parallel sorting.
 */
void sort_records(
    unsigned char * buffer,
    size_t record_size,
    size_t num_records,
    unsigned int num_threads,
    ThreadPool * thread_pool)
{
    if (num_threads > 1 && thread_pool != NULL) {
        quicksort_records_parallel(
            buffer,
            record_size,
            num_records,
            num_threads,
            thread_pool);
    } else {
        quicksort_records(buffer, record_size, num_records);
    }
}


/**
 * Remove duplicate records from an already sorted array.
 *
 * @return the number of unique records
 */
size_t filter_duplicate_records(
    unsigned char * buffer,
    unsigned int record_size,
    size_t num_records)
{
    // Special case for 0 or 1 records.
    if (num_records < 2) {
        return num_records;
    }

    // Find the first duplicate record.
    unsigned char * last_unique = buffer;
    unsigned char * next_record = buffer + record_size;
    size_t next_pos = 1;

    while (next_pos < num_records) {
        if (record_compare(last_unique, next_record, record_size) == 0) {
            break;
        }
        last_unique = next_record;
        next_record += record_size;
        next_pos++;
    }

    // Scan the rest of the records and copy unique records.
    size_t num_unique = next_pos;
    while (next_pos < num_records) {
        if (record_compare(last_unique, next_record, record_size) != 0) {
            num_unique++;
            last_unique += record_size;
            record_copy(last_unique, next_record, record_size);
        }
        next_record += record_size;
        next_pos++;
    }

    return num_unique;
}


/**
 * Sort the whole file in a single pass.
 *
 * This function is used only when the input file fits in memory.
 */
void single_pass(
    BinaryFile& input_file,
    BinaryFile& output_file,
    const SortContext& ctx)
{
    assert(input_file.size() < SIZE_MAX);

    size_t input_size = input_file.size();
    size_t num_records = input_size / ctx.record_size;

    log(ctx, "sorting %zu records in a single pass\n", num_records);

    // Allocate memory.
    log(ctx, "allocating memory\n");
    std::vector<unsigned char> buffer(input_size);

    Timer timer;

    // Read input file into memory.
    log(ctx, "reading input file\n");
    timer.start();
    input_file.read(buffer.data(), 0, input_size);
    timer.stop();
    log(ctx, "  t = %.3f seconds\n", timer.value());

    // Create thread pool for parallel sorting.
    std::unique_ptr<ThreadPool> thread_pool;
    if (ctx.num_threads > 1) {
        log(ctx, "creating thread pool with %u threads\n", ctx.num_threads);
        thread_pool.reset(new ThreadPool(ctx.num_threads));
    }

    // Sort records in memory buffer.
    log(ctx, "sorting records using %u threads\n", ctx.num_threads);
    timer.start();
    sort_records(
        buffer.data(),
        ctx.record_size,
        num_records,
        ctx.num_threads,
        thread_pool.get());
    timer.stop();
    log(ctx, "  t = %.3f seconds\n", timer.value());

    if (ctx.flag_unique) {
        // Eliminate duplicate records.
        log(ctx, "filtering duplicate records\n");
        timer.start();
        num_records = filter_duplicate_records(
            buffer.data(), ctx.record_size, num_records);
        timer.stop();
        log(ctx, "  t = %.3f seconds\n", timer.value());
        log(ctx, "found %zu unique records\n", num_records);
    }

    log(ctx, "writing output file\n");
    timer.start();

    // Shrink output file if duplicate records were removed.
    uint64_t output_size = num_records * ctx.record_size;
    if (output_size < input_size) {
        output_file.truncate_file(output_size);
    }

    // Write memory buffer to output file.
    output_file.write(buffer.data(), 0, output_size);
    timer.stop();
    log(ctx, "  t = %.3f seconds\n", timer.value());
}


/**
 * Perform the initial sort pass of multi-pass external sorting.
 *
 * All blocks will have the specified number of records, except for
 * the last block in the file which may be smaller.
 *
 * @param input_file            Input file.
 * @param output_file1          Output file for this pass.
 * @param output_file2          Second output file for this pass.
 * @param records_per_block     Number of records per sort block.
 * @param num_blocks            Number of sort blocks.
 * @param num_blocks_file1      Number of blocks for the first output file.
 * @param io_pool               Optional thread pool for background I/O.
 * @param ctx                   Reference to context structure.
 */
void sort_pass(
    BinaryFile& input_file,
    BinaryFile& output_file1,
    BinaryFile& output_file2,
    uint64_t records_per_block,
    uint64_t num_blocks,
    uint64_t num_blocks_file1,
    ThreadPool * io_pool,
    const SortContext& ctx)
{
    unsigned int record_size = ctx.record_size;
    uint64_t file_size = input_file.size();
    uint64_t num_records = file_size / record_size;

    log(ctx, "running initial sort pass\n");

    Timer timer;
    timer.start();

    // Create thread pool for parallel sorting.
    std::unique_ptr<ThreadPool> thread_pool;
    if (ctx.num_threads > 1) {
        log(ctx, "creating thread pool with %u threads\n", ctx.num_threads);
        thread_pool.reset(new ThreadPool(ctx.num_threads));
    }

    assert(records_per_block < SIZE_MAX / record_size);

    // Allocate sort buffer(s).
    std::array<std::vector<unsigned char>, 2> buffers;
    buffers[0].resize(records_per_block * record_size);
    if (io_pool != NULL) {
        // Allocate a second buffer for background I/O.
        buffers[1].resize(records_per_block * record_size);
    }

    // Helper function that returns the output file for a specified block.
    // This function deals with the nasty complication that the first bunch
    // of blocks must be written to a separate output file.
    auto output_file_chooser =
        [&output_file1, &output_file2, num_blocks_file1]
        (uint64_t block_index) -> BinaryFile& {
            if (block_index < num_blocks_file1) {
                return output_file1;
            } else {
                return output_file2;
            }
        };

    // Helper function that returns the buffer for a specified block.
    // This function deals with the nasty complication that we may or may not
    // be using two separate buffers for alternating blocks.
    auto buffer_chooser =
        [io_pool, &buffers]
        (uint64_t block_index) -> unsigned char * {
            if (io_pool != NULL) {
                // Blocks alternate between the two buffers.
                return buffers[block_index % 2].data();
            } else {
                // All blocks use the same buffer.
                return buffers[0].data();
            }
        };

    // Helper function to read a block from file to buffer.
    // This function deals with block offset calculations and with
    // the nasty complication that the last block may be shorter.
    auto read_block =
        [records_per_block, record_size, file_size]
        (BinaryFile& input_file, uint64_t block_index, unsigned char * buf) {
            uint64_t block_offset =
                block_index * records_per_block * record_size;
            size_t block_size = records_per_block * record_size;
            if (block_size > file_size - block_offset) {
                block_size = file_size - block_offset;
            }
            input_file.read(buf, block_offset, block_size);
        };

    // Helper function to write a block from buffer to file.
    // This function deals with block offset calculations and with
    // the nasty complication that the last block may be shorter.
    auto write_block =
        [records_per_block, record_size, file_size]
        (BinaryFile& output_file, uint64_t block_index, unsigned char * buf) {
            uint64_t block_offset =
                block_index * records_per_block * record_size;
            size_t block_size = records_per_block * record_size;
            if (block_size > file_size - block_offset) {
                block_size = file_size - block_offset;
            }
            output_file.write(buf, block_offset, block_size);
        };

    // Future object to wait for background I/O.
    ThreadPool::FutureType io_future;

    // Loop over blocks to be sorted.
    for (uint64_t block_index = 0; block_index < num_blocks; block_index++) {

        // Determine number of records in this block
        // (the last block may be shorter).
        uint64_t block_start_idx = block_index * records_per_block;
        size_t   block_num_records =
            (records_per_block < num_records - block_start_idx) ?
                records_per_block : (num_records - block_start_idx);

        log(ctx,
            "  sorting block %" PRIu64 " / %" PRIu64 ": %" PRIu64 " records\n",
            block_index,
            num_blocks,
            block_num_records);

        // Choose a buffer to sort this block.
        unsigned char * block_buffer = buffer_chooser(block_index);

        // Read block via blocking I/O, if necessary.
        // Use this to read all blocks when we do not use background I/O.
        // Use this only for the first block when we have background I/O.
        if (io_pool == NULL || block_index == 0) {
            read_block(input_file, block_index, block_buffer);
        }

        // Handle background I/O, if necessary.
        if (io_pool != NULL) {
            ThreadPool::FutureType new_future;
            if (block_index > 0 && block_index < num_blocks - 1) {
                // Start background I/O to write the previous block, then
                // read the next block when writing has finished.
                new_future = io_pool->submit(
                    [&input_file, &output_file_chooser, &buffer_chooser,
                     &read_block, &write_block, block_index]
                    () {
                        write_block(
                            output_file_chooser(block_index - 1),
                            block_index - 1,
                            buffer_chooser(block_index - 1));
                        read_block(
                            input_file,
                            block_index + 1,
                            buffer_chooser(block_index + 1));
                    });
            } else if (block_index < num_blocks - 1) {
                // Start background I/O to read next block.
                new_future = io_pool->submit(
                    [&input_file, &buffer_chooser, &read_block, block_index]
                    () {
                        read_block(
                            input_file,
                            block_index + 1,
                            buffer_chooser(block_index + 1));
                    });
            } else if (block_index > 0) {
                // Start background I/O to write previous block.
                new_future = io_pool->submit(
                    [&output_file_chooser, &buffer_chooser, &write_block,
                     block_index]
                    () {
                        write_block(
                            output_file_chooser(block_index - 1),
                            block_index - 1,
                            buffer_chooser(block_index - 1));
                    });
            }

            // Wait for completion of previous background I/O, if any.
            if (block_index > 0) {
                io_future.get();
            }

            // Keep future for the new background I/O, if any.
            io_future = std::move(new_future);
        }

        // Sort records in this block in-place.
        sort_records(
            block_buffer,
            record_size,
            block_num_records,
            ctx.num_threads,
            thread_pool.get());

        // If we use background I/O and this is the last block, wait for
        // any previous background write to complete.
        if (io_pool != NULL && block_index == num_blocks - 1) {
            io_future.get();
        }

        // Write block via blocking I/O, if necessary.
        // Use this to write all blocks when we do not use background I/O.
        // Use this only for the last block when we have background I/O.
        if (io_pool == NULL || block_index == num_blocks - 1) {
            BinaryFile& output_file = output_file_chooser(block_index);
            write_block(output_file, block_index, block_buffer);
        }
    }

    timer.stop();
    log(ctx, "end initial sort pass\n");
    log(ctx, "  t = %.3f seconds\n", timer.value());
}


/**
 * Merge sorted blocks of records into a single sorted block.
 *
 * @param input_streams     One input stream for each input blocks.
 * @param output_stream     Output stream for the merged block.
 * @param record_size       Record size in bytes.
 * @param branch_factor     Number of input blocks.
 *                          May be less than the length of input_streams.
 * @param filter_dupl       True to eliminate duplicate records.
 */
void merge_blocks(
    std::vector<std::unique_ptr<RecordInputStream>>& input_streams,
    RecordOutputStream& output_stream,
    size_t record_size,
    unsigned int branch_factor,
    bool filter_dupl)
{
    assert(branch_factor > 1);
    assert(branch_factor <= input_streams.size());

    // Put the head element of each block into a heap.
    // The heap will determine which block contains the element that
    // should go first in the merged block.
    typedef std::tuple<const unsigned char*, RecordInputStream*> HeapElement;

    // Function which compares records and returns true if
    // record A comes after record B in sort order.
    // If this function is used as the compare operator of a max-heap,
    // the record that comes first in sort order will be at the top
    // of the heap.
    auto cmp_heap_elem =
        [record_size](const HeapElement& a, const HeapElement& b) {
            const unsigned char *reca = std::get<0>(a);
            const unsigned char *recb = std::get<0>(b);
            return record_compare(reca, recb, record_size) > 0;
        };

    // Initialize empty heap.
    std::vector<HeapElement> heap;

    // Get the first element of each block.
    for (unsigned int i = 0; i < branch_factor; i++) {
        // Input blocks should not be empty.
        assert(!input_streams[i]->empty());
        heap.emplace_back(input_streams[i]->record(), input_streams[i].get());
    }

    // Make a heap of the first blocks.
    std::make_heap(heap.begin(), heap.end(), cmp_heap_elem);

    // Allocate a temporary record for duplicate filtering.
    std::vector<unsigned char> temp_record(record_size);

    // The very first record can not be filtered out.
    bool filter_first_pass = true;

    // Keep merging until the heap runs empty.
    while (!heap.empty()) {

        // Extract the first element from the heap.
        const unsigned char * rec;
        RecordInputStream * instream;
        std::tie(rec, instream) = heap[0];
        std::pop_heap(heap.begin(), heap.end(), cmp_heap_elem);

        if (filter_dupl) {

            // Compare against previous record, only output if different.
            if (filter_first_pass
                || record_compare(temp_record.data(), rec, record_size) != 0)
            {
                output_stream.put(rec);
                record_copy(temp_record.data(), rec, record_size);
            }
            filter_first_pass = false;

        } else {

            // No filtering, just push record to the output block.
            output_stream.put(rec);

        }

        // Try to pull the next record from this input stream.
        instream->next_record();
        if (instream->empty()) {
            // Stream is empty. This stream is now out of the game.
            // The heap shrinks by 1 element.
            heap.pop_back();
        } else {
            // Push next record from the stream into the heap.
            heap.back() = std::make_tuple(instream->record(), instream);
            std::push_heap(heap.begin(), heap.end(), cmp_heap_elem);
        }
    }
}


/**
 * Perform a merge pass of multi-pass external sorting.
 *
 * All input blocks will have the specified number of records, except for
 * the last block in the file which may be smaller.
 *
 * All output blocks will have (branch_factor * records_per_block) records,
 * except for the last output block in the file which may be smaller.
 *
 * If "filter_duplicates" is specified, the number of output blocks MUST be 1.
 * In this case duplicate elements will be removed from the output.
 *
 * @param input_file            Input file for this pass.
 * @param output_file           Output file for this pass.
 * @param records_per_block     Number of records per input block.
 * @param num_blocks            Number of input blocks.
 * @param branch_factor         Number of blocks to merge per output block.
 * @param filter_dupl           True to eliminate duplicate records.
 * @param read_thread_pool      Optional thread pool for background read I/O.
 * @param write_thread_pool     Optional thread pool for background write I/O.
 * @param ctx                   Reference to context structure.
 */
void merge_pass(
    BinaryFile& input_file,
    BinaryFile& output_file,
    const SortStrategy::MergePass& merge_pass,
    bool filter_dupl,
    ThreadPool * read_thread_pool,
    ThreadPool * write_thread_pool,
    const SortContext& ctx)
{
    size_t num_blocks = merge_pass.records_per_block.size();

    // Only filter duplicates when the output is a single block.
    assert((!filter_dupl) || num_blocks <= ctx.branch_factor);

    Timer timer;
    timer.start();

    // Choose number of buffers.
    // Without background I/O: 1 input buffer per stream + 1 output buffer.
    // With background I/O:    2 input buffers per stream + 4 output buffers.
    unsigned int num_input_buf = (read_thread_pool != NULL) ? 2 : 1;
    unsigned int num_output_buf = (write_thread_pool != NULL) ? 4 : 1;
    unsigned int num_buffers =
        ctx.branch_factor * num_input_buf + num_output_buf;

    // Calculate buffer size.
    // Must not exceed the available memory.
    // Must be a multiple of the record size and the transfer alignment.
    // Should not exceed MAX_MERGE_BUFFER_SIZE.
    size_t buffer_alignment = TRANSFER_ALIGNMENT * size_t(ctx.record_size);
    size_t buffer_size =
        std::min(ctx.memory_size / num_buffers,
                 std::max(buffer_alignment, size_t(MAX_MERGE_BUFFER_SIZE)));
    buffer_size -= buffer_size % buffer_alignment;

    // Prepare a list of blocks for each input stream.
    std::vector<RecordInputStream::BlockList> stream_blocks(ctx.branch_factor);
    uint64_t file_offset = 0;
    for (size_t p = 0; p < num_blocks; p++) {
        uint64_t num_records = merge_pass.records_per_block[p];
        unsigned int streamidx = p % ctx.branch_factor;
        stream_blocks[streamidx].emplace_back(file_offset, num_records);
        file_offset += num_records * ctx.record_size;
    }

    // Initialize input streams.
    std::vector<std::unique_ptr<RecordInputStream>> input_streams;
    for (unsigned int i = 0; i < ctx.branch_factor; i++) {
        input_streams.emplace_back(new RecordInputStream(
            input_file,
            ctx.record_size,
            std::move(stream_blocks[i]),
            buffer_size,
            read_thread_pool));
    }

    // Initialize output stream.
    RecordOutputStream output_stream(
        output_file,
        ctx.record_size,
        0,
        buffer_size,
        num_output_buf,
        write_thread_pool);

    // Loop over groups of blocks to be sorted.
    // Every group consists of "branch_factor" blocks, except the last
    // group which may contain fewer blocks.
    // Each group produces one output block.
    size_t block_index = 0;
    while (block_index < num_blocks) {

        // If this is the first merge pass, the last group may have
        // fewer than "branch_factor" blocks.
        unsigned int blocks_this_group = ctx.branch_factor;
        if (blocks_this_group > num_blocks - block_index) {
            blocks_this_group = num_blocks - block_index;
        }

        // Merging a single block with itself would be dumb.
        // And our strategy planner is not that dumb.
        assert(blocks_this_group > 1);

        // Skip to the next section of each active input stream.
        for (unsigned int i = 0; i < blocks_this_group; i++) {
            input_streams[i]->next_block();
        }

        // Merge the blocks.
        merge_blocks(
            input_streams,
            output_stream,
            ctx.record_size,
            blocks_this_group,
            filter_dupl);

        // Skip to the start of the next block group.
        block_index += blocks_this_group;
    }

    // Flush output stream buffers.
    output_stream.flush();

    // Shrink output file if duplicate records were removed.
    if (filter_dupl) {
        uint64_t output_size = output_stream.file_offset();
        uint64_t num_output_records = output_size / ctx.record_size;
        log(ctx, "found %zu unique records\n", num_output_records);
        output_file.truncate_file(output_size);
    }

    timer.stop();
    log(ctx, "  t = %.3f seconds\n", timer.value());
}


/**
 * Prepare a strategy for multi-pass external sorting.
 */
SortStrategy plan_multi_pass_strategy(
    uint64_t file_size,
    const SortContext& ctx)
{
    // Plan the initial sort pass.

    // When background I/O is enabled, use blocks that are at most half of
    // available memory so we can use two buffers to overlap I/O and sorting.
    uint64_t max_sort_block_size =
        ctx.flag_io_thread ? (ctx.memory_size / 2) : ctx.memory_size;

    // Calculate number of records per block.
    // Make sure this is a multiple of the transfer alignment size.
    uint64_t records_per_sort_block = max_sort_block_size / ctx.record_size;
    records_per_sort_block -= records_per_sort_block % TRANSFER_ALIGNMENT;

    // Calculate number of blocks during the initial sort pass.
    uint64_t num_records = file_size / ctx.record_size;
    uint64_t num_sort_blocks = 1 + (num_records - 1) / records_per_sort_block;

    // A list of blocks is constructed in memory for each merge pass.
    // There are several of these lists. Together they will consume
    // about 40 bytes per block. If we get an insanely large input file
    // with a small memory limit, this metadata could by itself consume
    // too much memory.
    // Let's do a sanity check to ensure that the metadata uses less than
    // 25% of the memory limit.
    if (num_sort_blocks >= ctx.memory_size / 4 / 40) {
        throw std::logic_error(
            "Not enough memory to manage the list of blocks");
    }

    // Plan the merge passes.
    //
    // In prinicple, every pass merges groups of "branch_factor" input blocks
    // into one output block per group, thus reducing the number of remaining
    // blocks by a factor "branch_factor".
    //
    // However, this gets more complicated if the merge tree is not perfectly
    // balanced, which happens if the number of sort blocks is not a power
    // of "branch_factor". In that case, the first merge pass will have
    // to make things right by handling only a subset of the data.
    //
    // The first merge pass processes a subset of the sort blocks.
    // It merges groups of "branch_factor" sort blocks into one output block
    // per group. The last group in this pass may contain fewer than
    // "branch_factor" sort blocks. After the first merge pass, the number
    // of remaining blocks is an exact power of "branch_factor".
    // The remaining blocks are in general not all the same size.
    //
    // After the first merge pass, each subsequent pass (if any) merges
    // groups of exactly "branch_factor" blocks into one output block per
    // group. These blocks are in general not all the same size.
    //
    // Example:
    //
    // branch_factor = 3
    // num_sort_blocks = 12
    //
    // Sorted blocks:
    // [S00] [S01] [S02] [S03] [S04] [S05] [S06] [S07] [S08] [S09] [S10] [S11]
    //
    // There are 3 merge passes.
    // The first merge pass handles only sort blocks S00 - S04.
    // The next two merge passes handle groups of exactly 3 blocks.
    //
    // [S00] [S01] [S02] [S03] [S04] [S05] [S06] [S07] [S08] [S09] [S10] [S11]
    //   |     |     |     |     |     |     |     |     |     |     |     |
    //   +-----+-----+     +--+--+     |     |     |     |     |     |     |
    //         |              |        |     |     |     |     |     |     |
    //      [S00-S02]     [S03-S04]  [S05] [S06] [S07] [S08] [S09] [S10] [S11]
    //            |           |        |     |     |     |     |     |     |
    //            +-----------+--------+     +-----+-----+     +-----+-----+
    //                        |                    |                 |
    //                    [S00-S05]            [S06-S08]         [S09-S11]
    //                        |                    |                 |
    //                        +--------------------+-----------------+
    //                                             |
    //                                         [S00-S11]
    //
    // Note that a subset of sort blocks enter into the first merge pass
    // while the remaining sort blocks go directly into the second merge pass.
    // (It is also possible that all sort blocks go into the first pass,
    // if the merge tree is perfectly balanced.)
    //

    // Determine the number of full merge passes (2nd pass and later).
    unsigned int num_merge_pass = 0;
    uint64_t num_merge_blocks = 1;
    while (num_merge_blocks * ctx.branch_factor < num_sort_blocks) {
        num_merge_blocks *= ctx.branch_factor;
        num_merge_pass++;
    }

    // Add a first merge pass.
    num_merge_pass++;

    // Determine the number of merge groups in the first merge pass.
    // The last group may have fewer than "branch_factor" input blocks.
    uint64_t num_merge_ops_first_pass =
        (num_sort_blocks - num_merge_blocks + (ctx.branch_factor - 1) - 1)
        / (ctx.branch_factor - 1);

    // Determine the number of sort blocks to process in the first merge pass.
    uint64_t num_sort_blocks_first_merge =
        num_sort_blocks - num_merge_blocks + num_merge_ops_first_pass;

    assert(num_sort_blocks_first_merge <= num_sort_blocks);

    SortStrategy strategy;
    strategy.records_per_sort_block = records_per_sort_block;
    strategy.num_sort_blocks = num_sort_blocks;
    strategy.num_sort_blocks_first_merge = num_sort_blocks_first_merge;

    // Plan the details of each merge pass.
    //
    // The first merge pass handles a subset of the sort blocks.
    // All of these sort blocks are the same size, except possibly the
    // last block if it runs to the end of the file.
    {
        strategy.merge_pass.emplace_back();
        SortStrategy::MergePass& merge_pass = strategy.merge_pass.back();
        for (size_t i = 0; i < num_sort_blocks_first_merge; i++) {
            uint64_t records_this_block =
                std::min(records_per_sort_block,
                         num_records - i * records_per_sort_block);
            merge_pass.records_per_block.push_back(records_this_block);
        }
    }

    // Plan the rest of the passes.
    for (unsigned int mp = 1; mp < num_merge_pass; mp++) {

        strategy.merge_pass.emplace_back();
        SortStrategy::MergePass& merge_pass = strategy.merge_pass.back();
        SortStrategy::MergePass& prev_pass = *(strategy.merge_pass.end() - 2);

        uint64_t records_this_pass = 0;

        // Output from the previous pass is input into this pass.
        uint64_t records_this_block = 0;
        for (size_t i = 0; i < prev_pass.records_per_block.size(); i++) {
            records_this_block += prev_pass.records_per_block[i];
            if ((i + 1) % ctx.branch_factor == 0) {
                merge_pass.records_per_block.push_back(records_this_block);
                records_this_pass += records_this_block;
                records_this_block = 0;
            }
        }

        // The last group of the first pass may merge fewer than
        // "branch_factor" blocks.
        if (records_this_block > 0) {
            merge_pass.records_per_block.push_back(records_this_block);
            records_this_pass += records_this_block;
        }

        if (mp == 1) {
            // The second pass handles sort blocks that were skipped
            // during the first pass.
            for (size_t i = num_sort_blocks_first_merge;
                 i < num_sort_blocks;
                 i++) {
                records_this_block = std::min(
                    records_per_sort_block,
                    num_records - i * records_per_sort_block);
                merge_pass.records_per_block.push_back(records_this_block);
                records_this_pass += records_this_block;
            }
        }

        // Check that number of input blocks is divisible by branch_factor.
        assert(merge_pass.records_per_block.size() % ctx.branch_factor == 0);

        // Check that all records are accounted for.
        assert(records_this_pass == num_records);
    }

    // Double-check that the last pass produces a single output block.
    SortStrategy::MergePass& last_pass = strategy.merge_pass.back();
    assert(last_pass.records_per_block.size() <= ctx.branch_factor);

    return strategy;
}


/**
 * Sort a binary data file.
 *
 * @param input_file    Path name of binary input file.
 * @param output_file   Path name of binary output file.
 * @param ctx           Reference to context structure.
 */
void sortbin(
    const std::string& input_name,
    const std::string& output_name,
    const SortContext& ctx)
{
    log(ctx, "using memory_size = %" PRIu64 " bytes\n", ctx.memory_size);

    if (ctx.branch_factor < 2) {
        throw std::logic_error("Invalid branch factor (must be >= 2)");
    }
    if (ctx.branch_factor > MAX_BRANCH_FACTOR) {
        throw std::logic_error(
            "Invalid branch factor (must be <= "
            + std::to_string(MAX_BRANCH_FACTOR)
            + ")");
    }

    // We want file I/O to occur on 4096-byte boundaries.
    // To ensure this, we want to do I/O on multiples of 4096 records.
    // To ensure this is possible, we need room for ~ 32k records per branch.
    if (ctx.memory_size / ctx.record_size / ctx.branch_factor <
        8 * TRANSFER_ALIGNMENT) {
        throw std::logic_error(
            "Not enough memory for this combination of record size"
            " and branch factor");
    }

    // Open input file.
    log(ctx, "opening input file\n");
    BinaryInputFile input_file(input_name);
    uint64_t file_size = input_file.size();

    // Check that input file contains an integer number of records.
    if (file_size % ctx.record_size != 0) {
        throw std::logic_error(
            "Input file does not contain an integer number of records");
    }

    // Create output file.
    log(ctx, "creating output file\n");
    BinaryOutputFile output_file(output_name, file_size);

    if (file_size <= ctx.memory_size) {
        // Data fits in memory.

        // Sort in a single pass.
        single_pass(input_file, output_file, ctx);

    } else {
        // Data does not fit in memory.

        // Plan a multi-pass strategy.
        SortStrategy strategy = plan_multi_pass_strategy(file_size, ctx);
        unsigned int num_merge_pass = strategy.merge_pass.size();

        log(ctx,
            "sorting %" PRIu64 " records in %" PRIu64 " blocks"
            " followed by %u merge passes\n",
            file_size / ctx.record_size,
            strategy.num_sort_blocks,
            num_merge_pass);

        // Create a temporary file.
        log(ctx, "creating temporary file\n");
        BinaryTempFile temp_file(ctx.temporary_directory, file_size);

        // Create thread pools for background I/O.
        std::array<std::unique_ptr<ThreadPool>, 2> io_thread_pool;
        if (ctx.flag_io_thread) {
            // Create separate threads for reading and writing.
            // Just 1 thread per pool because
            //  - More than 1 write thread is useless because there is
            //    only one sequential output stream.
            //  - More than 1 read thread may increase seeking while
            //    in fact we want to encourage big sequential transfers.
            log(ctx, "creating background I/O threads\n");
            io_thread_pool[0].reset(new ThreadPool(1));
            io_thread_pool[1].reset(new ThreadPool(1));
        }

        // The merge passes alternate between tempfile-to-outputfile and
        // outputfile-to-tempfile.
        BinaryFile * output_or_temp_file[2] = { &output_file, &temp_file };

        // The final merge pass will be tempfile-to-outputfile.
        // Depending on the number of merge passes, the first merge pass
        // reads either from the output file or from the tempfile.
        //
        // The sort pass feeds blocks into the first merge pass,
        // but may also feed blocks into the second merge pass if the
        // merge tree is unbalanced.
        {
            BinaryFile * sort_output_first_merge_pass =
                output_or_temp_file[num_merge_pass % 2];
            BinaryFile * sort_output_second_merge_pass =
                output_or_temp_file[(num_merge_pass - 1) % 2];

            // Execute the initial sort pass.
            sort_pass(
                input_file,
                *sort_output_first_merge_pass,
                *sort_output_second_merge_pass,
                strategy.records_per_sort_block,
                strategy.num_sort_blocks,
                strategy.num_sort_blocks_first_merge,
                io_thread_pool[0].get(),
                ctx);
        }

        // Execute the merge passes.
        for (unsigned int mp = 0; mp < num_merge_pass; mp++) {
            const SortStrategy::MergePass& mpass = strategy.merge_pass[mp];
            size_t num_blocks = mpass.records_per_block.size();
            uint64_t num_records = std::accumulate(
                mpass.records_per_block.begin(),
                mpass.records_per_block.end(),
                uint64_t(0));

            log(ctx,
                "running merge pass %u / %u:"
                " %zu blocks, %" PRIu64 " records\n",
                mp,
                num_merge_pass,
                num_blocks,
                num_records);

            // Filter duplicate records only on the last pass.
            bool filter_dupl = ctx.flag_unique && (mp + 1 == num_merge_pass);

            // Alternate between temp_file and output_file.
            BinaryFile * pass_input_file =
                output_or_temp_file[(num_merge_pass - mp) % 2];
            BinaryFile * pass_output_file =
                output_or_temp_file[(num_merge_pass - mp - 1) % 2];

            // Execute the merge pass.
            merge_pass(
                *pass_input_file,
                *pass_output_file,
                strategy.merge_pass[mp],
                filter_dupl,
                io_thread_pool[0].get(),
                io_thread_pool[1].get(),
                ctx);
        }
    }

    log(ctx, "finished\n");
}


std::string get_default_tmpdir(void)
{
    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir == NULL) {
        tmpdir = "/tmp";
    }
    return std::string(tmpdir);
}


/** Parse an unsigned integer. */
bool parse_uint(const char *argstr, unsigned int& value)
{
    char *endptr;

    errno = 0;
    long t = strtol(argstr, &endptr, 10);

    if (endptr == argstr || endptr[0] != '\0') {
        return false;
    }

    if (errno != 0) {
        return false;
    }

    if (t < 0 || (unsigned long)t > UINT_MAX) {
        return false;
    }

    value = t;
    return true;
}


/**
 * Parse a memory size specification.
 *
 * Memory size must be specified as an integer with suffix "M" or "G".
 *
 * @return specfified memory size in bytes, or 0 for an invalid size
 */
uint64_t parse_memory_size(const char *argstr)
{
    char *endptr;
    errno = 0;

    long long value = strtoll(argstr, &endptr, 10);
    if (endptr == argstr
            || (endptr[0] != 'G' && endptr[0] != 'M') || endptr[1] != '\0') {
        fprintf(stderr,
                "ERROR: Invalid memory size."
                " Specify e.g. '--memory=800M' or '--memory=4G'.\n");
        return 0;
    }

    if (value <= 0 || errno != 0) {
        fprintf(stderr, "ERROR: Invalid memory size\n");
        return 0;
    }

    uint64_t factor;
    if (endptr[0] == 'G') {
        factor = 1024 * 1024 * 1024;
    } else {
        factor = 1024 * 1024;
    }

    if ((unsigned long long)value > UINT64_MAX / factor) {
        fprintf(stderr, "ERROR: Invalid memory size\n");
        return 0;
    }

    return value * factor;
}


void usage()
{
    fprintf(stderr,
        "\n"
        "Sort fixed-length binary records.\n"
        "\n"
        "Usage: sortbin [options] inputfile outputfile\n"
        "\n"
        "Options:\n"
        "\n"
        "  -s, --size=N\n"
        "        Specify record size of N bytes (required)\n"
        "\n"
        "  -u, --unique\n"
        "        Eliminate duplicates after sorting\n"
        "\n"
        "  --memory=<n>M, --memory=<n>G\n"
        "        Specify the amount of RAM that may be used.\n"
        "        Use suffix 'M' for MiByte, or 'G' for GiByte."
        " (default: %d MiB)\n"
        "\n"
        "  --branch=N\n"
        "        Merge at most N subarrays in one step. (default: %d)\n"
        "\n"
        "  --parallel=N\n"
        "        Use N threads for parallel sorting. (default: %d)\n"
        "\n"
        "  --iothread / --no-iothread\n"
        "        Enable or disable use of background threads for I/O.\n"
        "        (default: %s)\n"
        "\n"
        "  -T, --temporary-directory=DIR\n"
        "        Write temporary file to the specified directory."
        " (default: $TMPDIR)\n"
        "\n"
        "  -v, --verbose\n"
        "        Write progress messages to STDERR.\n"
        "\n"
        "The output file must not yet exist.\n"
        "If the file does not fit in memory, a temporary file will be\n"
        "created with the same size as the input/output files.\n"
        "\n",
        DEFAULT_MEMORY_SIZE_MBYTE,
        DEFAULT_BRANCH_FACTOR,
        DEFAULT_THREADS,
        DEFAULT_IO_THREAD ? "enable" : "disable");
}


} // anonymous namespace


int main(int argc, char **argv)
{
    const struct option longopts[] = {
        { "size", 1, NULL, 's' },
        { "unique", 0, NULL, 'u' },
        { "memory", 1, NULL, 'M' },
        { "branch", 1, NULL, 'B' },
        { "parallel", 1, NULL, 'P' },
        { "temporary-directory", 1, NULL, 'T' },
        { "iothread", 0, NULL, 'X' },
        { "no-iothread", 0, NULL, 'x' },
        { "verbose", 0, NULL, 'v' },
        { "help", 0, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    SortContext ctx;
    ctx.record_size = 0;
    ctx.memory_size = uint64_t(DEFAULT_MEMORY_SIZE_MBYTE) * 1024 * 1024;
    ctx.branch_factor = DEFAULT_BRANCH_FACTOR;
    ctx.num_threads = DEFAULT_THREADS;
    ctx.flag_unique = false;
    ctx.flag_io_thread = DEFAULT_IO_THREAD;
    ctx.flag_verbose = false;
    ctx.temporary_directory = get_default_tmpdir();

    int opt;
    while ((opt = getopt_long(argc, argv, "s:T:uvh", longopts, NULL)) != -1) {
        switch (opt) {
            case 's':
                if (!parse_uint(optarg, ctx.record_size)
                        || ctx.record_size < 1) {
                    fprintf(stderr,
                        "ERROR: Invalid record size (must be at least 1)\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'u':
                ctx.flag_unique = true;
                break;
            case 'M':
                ctx.memory_size = parse_memory_size(optarg);
                if (ctx.memory_size == 0) {
                    return EXIT_FAILURE;
                }
                break;
            case 'B':
                if (!parse_uint(optarg, ctx.branch_factor)
                        || ctx.branch_factor < 2) {
                    fprintf(stderr,
                        "ERROR: Invalid branch factor, must be at least 2\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'P':
                if (!parse_uint(optarg, ctx.num_threads)
                        || ctx.num_threads < 1
                        || ctx.num_threads > MAX_THREADS) {
                    fprintf(stderr, "ERROR: Invalid number of threads\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'T':
                ctx.temporary_directory = optarg;
                break;
            case 'X':
                ctx.flag_io_thread = true;
                break;
            case 'x':
                ctx.flag_io_thread = false;
                break;
            case 'v':
                ctx.flag_verbose = true;
                break;
            case 'h':
                usage();
                return EXIT_SUCCESS;
            default:
                usage();
                return EXIT_FAILURE;
        }
    }

    if (ctx.record_size < 1) {
        fprintf(stderr, "ERROR: Missing required parameter --size\n");
        usage();
        return EXIT_FAILURE;
    }

    if (argc < optind + 2) {
        fprintf(stderr,
                "ERROR: Input and output file names must be specified\n");
        usage();
        return EXIT_FAILURE;
    }

    if (argc > optind + 2) {
        fprintf(stderr, "ERROR: Unexpected command-line parameters\n");
        usage();
        return EXIT_FAILURE;
    }

    if (ctx.memory_size >= SIZE_MAX) {
        fprintf(
            stderr,
            "ERROR: This system supports at most %zu MB memory\n",
            SIZE_MAX / 1024 / 1024 - 1);
        return EXIT_FAILURE;
    }

    std::string input_name(argv[optind]);
    std::string output_name(argv[optind+1]);

    try {
        sortbin(input_name, output_name, ctx);
    } catch (const std::exception& ex) {
        fprintf(stderr, "ERROR: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
