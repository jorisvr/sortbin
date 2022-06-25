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
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>


/* Maximum amount of RAM to use (in MBytes). */
#define DEFAULT_MEMORY_SIZE_MBYTE   1024

/* Default branch factor while merging. */
#define DEFAULT_BRANCH_FACTOR       16

/* Default number of sorting threads. */
#define DEFAULT_THREADS             1

/* Align buffer sizes and I/O on this number of records.
   For efficiency, I/O should be done in multiples of 4096 bytes. */
#define TRANSFER_ALIGNMENT          4096

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

    /** True to eliminate duplicate records. */
    bool flag_unique;

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

        /** Number of records per input block into this merge pass. */
        uint64_t records_per_input_block;

        /** Total number of input blocks into this merge pass. */
        uint64_t num_input_blocks;

        /** Number of input blocks to merge into each output block. */
        unsigned int branch_factor;
    };

    /** Number of records per block during the initial sort pass. */
    uint64_t records_per_sort_block;

    /** Number of blocks for the initial sort pass. */
    uint64_t num_sort_blocks;

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
 * The input stream reads from a sequence of discontinuous, equally spaced
 * blocks in the input file. All blocks have the same size, except for
 * the last block which may be shorter if it runs to the end of the file.
 * Each block contains a flat array of binary records.
 *
 * The input stream starts in the "empty" state.
 * The first call to "next_block()" enables reading records from the
 * first input block. When the input stream reaches the end of the
 * current block, it again enters the "empty" state until the following
 * call to "next_block()".
 */
class RecordInputStream
{
// TODO : double-buffering with delayed I/O via background thread
public:
    /**
     * Construct a record input stream.
     *
     * The stream will initially contain no input sections.
     *
     * @param input_file    Input file where records read from.
     * @param record_size   Record size in bytes.
     * @param start_offset  Offset in input file of first input section.
     * @param block_size    Size of each input block in bytes.
     *                      Must be a multiple of "record_size".
     *                      The last input block may be shorter if it runs
     *                      to the end of the file.
     * @param block_stride  Distance between start of blocks in bytes.
     * @param buffer_size   Buffer size in bytes.
     *                      Must be a multiple of "record_size".
     *                      Note: Each RecordInputStream creates two buffers
     *                      of the specified size.
     */
    RecordInputStream(
        BinaryFile&   input_file,
        unsigned int  record_size,
        uint64_t      start_offset,
        uint64_t      block_size,
        uint64_t      block_stride,
        size_t        buffer_size)
      : m_input_file(input_file),
        m_record_size(record_size),
        m_block_offset(start_offset),
        m_block_size(block_size),
        m_block_stride(block_stride),
        m_block_remaining(0),
        m_file_offset(0),
        m_bufpos(NULL),
        m_bufend(NULL),
        m_buffer(buffer_size)
    {
        assert(start_offset <= input_file.size());
        assert(block_size % record_size == 0);
        assert(block_size <= block_stride);
        assert(buffer_size % record_size == 0);
        assert(buffer_size > record_size);
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

        uint64_t file_size = m_input_file.size();
        assert(m_block_offset <= file_size);

        m_file_offset = m_block_offset;
        m_block_remaining =
            std::min(m_block_size, file_size - m_block_offset);

        m_block_offset += std::min(m_block_stride,
                                   file_size - m_block_offset);

        refill_buffer();
    }

private:
    /** Refill the buffer from the current input section. */
    void refill_buffer()
    {
        if (m_block_remaining > 0) {
            size_t transfer_size =
                (m_buffer.size() < m_block_remaining) ?
                    m_buffer.size() : m_block_remaining;
            m_input_file.read(m_buffer.data(), m_file_offset, transfer_size);
            m_file_offset += transfer_size;
            m_block_remaining -= transfer_size;
            m_bufpos = m_buffer.data();
            m_bufend = m_buffer.data() + transfer_size;
        }
    }

    BinaryFile&         m_input_file;
    const unsigned int  m_record_size;
    uint64_t            m_block_offset;
    uint64_t            m_block_size;
    uint64_t            m_block_stride;
    uint64_t            m_block_remaining;
    uint64_t            m_file_offset;
    unsigned char *     m_bufpos;
    unsigned char *     m_bufend;
    std::vector<unsigned char> m_buffer;
};


/**
 * Write binary records to an output file with buffering.
 */
class RecordOutputStream
{
// TODO : double-buffering with delayed I/O via background thread
public:
    /**
     * Construct a record output stream.
     *
     * @param output_file   Output file where records are written.
     * @param record_size   Record size in bytes.
     * @param file_offset   Start offset in the output file.
     * @param buffer_size   Buffer size in bytes.
     *                      Note: Each RecordOutputStream creates two buffers
     *                      of the specified size.
     */
    RecordOutputStream(
        BinaryFile&   output_file,
        unsigned int  record_size,
        uint64_t      file_offset,
        size_t        buffer_size)
      : m_output_file(output_file),
        m_record_size(record_size),
        m_file_offset(file_offset),
        m_buffer(buffer_size)
    {
        m_bufpos = m_buffer.data();
        m_bufend = m_buffer.data() + m_buffer.size();
    }

    // Prevent copying and assignment.
    RecordOutputStream(const RecordOutputStream&) = delete;
    RecordOutputStream& operator=(const RecordOutputStream&) = delete;

    /** Append a record to the output stream. */
    inline void put(const unsigned char *record)
    {
        if (m_record_size > m_bufend - m_bufpos) {
            flush();
        }
        memcpy(m_bufpos, record, m_record_size);
        m_bufpos += m_record_size;
    }

    /** Return the current file offset. Flush before calling this function. */
    inline uint64_t file_offset() const
    {
        assert(m_bufpos == m_buffer.data());
        return m_file_offset;
    }

    /** Flush buffered records to the output file. */
    void flush()
    {
        size_t flush_size = m_bufpos - m_buffer.data();
        if (flush_size > 0) {
            m_output_file.write(m_buffer.data(), m_file_offset, flush_size);
            m_file_offset += flush_size;
            m_bufpos = m_buffer.data();
        }
    }

private:
    BinaryFile&         m_output_file;
    const unsigned int  m_record_size;
    uint64_t            m_file_offset;
    unsigned char *     m_bufpos;
    unsigned char *     m_bufend;
    std::vector<unsigned char>  m_buffer;
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

        // Initialize pointers to start, end and middle of range.
        unsigned char * left_ptr = range_begin;
        unsigned char * right_ptr =
            range_begin + (range_num_records - 1) * record_size;
        unsigned char * pivot_ptr =
            range_begin + (range_num_records / 2) * record_size;

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

        // Push left subrange on the stack, if it meets the size threshold.
        size_t num_left =
            (right_ptr + record_size - range_begin) / record_size;
        if (num_left > insertion_sort_threshold) {
            stack.emplace_back(range_begin, num_left, depth_limit - 1);
        }

        // Push right subrange on the stack, if it meets the size threshold.
        size_t num_right =
            range_num_records - (left_ptr - range_begin) / record_size;
        if (num_right > insertion_sort_threshold) {
            stack.emplace_back(left_ptr, num_right, depth_limit - 1);
        }
    }

    // Recursive partitining finished.
    // The array is now roughly sorted, except for subranges that were
    // skipped because they are within the threshold.

    // Finish with insertion sort to get a fully sorted array.
    insertion_sort_records(buffer, record_size, num_records);
}


/** Sort the specified block of records (in-place). */
void sort_records(
    unsigned char * buffer,
    size_t record_size,
    size_t num_records)
{
    // TODO : multi-threaded quicksort

//    heap_sort_records(buffer, record_size, num_records);
    quicksort_records(buffer, record_size, num_records);
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
        if (memcmp(last_unique, next_record, record_size) == 0) {
            break;
        }
        last_unique = next_record;
        next_record += record_size;
        next_pos++;
    }

    // Scan the rest of the records and copy unique records.
    size_t num_unique = next_pos;
    while (next_pos < num_records) {
        if (memcmp(last_unique, next_record, record_size) != 0) {
            num_unique++;
            last_unique += record_size;
            memcpy(last_unique, next_record, record_size);
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

// TODO : multi-threaded sorting with thread pool

    // Sort records in memory buffer.
    log(ctx, "sorting records\n");
    timer.start();
    sort_records(buffer.data(), ctx.record_size, num_records);
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
 * @param output_file           Output file for this pass.
 * @param records_per_block     Number of records per sort block.
 * @param num_blocks            Number of sort blocks.
 * @param ctx                   Reference to context structure.
 */
void sort_pass(
    BinaryFile& input_file,
    BinaryFile& output_file,
    uint64_t records_per_block,
    uint64_t num_blocks,
    const SortContext& ctx)
{
    unsigned int record_size = ctx.record_size;
    uint64_t file_size = input_file.size();
    uint64_t num_records = file_size / record_size;

    log(ctx, "running initial sort pass\n");

    Timer timer;
    timer.start();

// TODO : double-buffer with I/O in separate thread
    // Allocate sort buffer.
    assert(records_per_block < SIZE_MAX / record_size);
    std::vector<unsigned char> buffer(records_per_block * record_size);

// TODO : multi-threaded sorting with thread pool

    // Loop over blocks to be sorted.
    for (uint64_t block_index = 0; block_index < num_blocks; block_index++) {

        uint64_t first_record_idx = block_index * records_per_block;
        size_t block_num_records =
            std::min(records_per_block, num_records - first_record_idx);

        log(ctx,
            "sorting block %" PRIu64 " / %" PRIu64 ": %" PRIu64 " records\n",
            block_index,
            num_blocks,
            block_num_records);

        // Read block.
        input_file.read(
            buffer.data(),
            first_record_idx * record_size,
            block_num_records * record_size);

        // Sort records in this block.
        sort_records(
            buffer.data(),
            record_size,
            block_num_records);

        // Write block.
        output_file.write(
            buffer.data(),
            first_record_idx * record_size,
            block_num_records * record_size);
    }

    timer.stop();
    log(ctx, "initial sort pass finished\n");
    log(ctx, "  t = %.3f seconds\n", timer.value());
}


/**
 * Merge 2 sorted blocks of records into a single sorted block.
 *
 * @param instream1         Input stream containing block 1.
 * @param instream2         Input stream containing block 2.
 * @param output_stream     Output stream for the merged block.
 * @param record_size       Record size in bytes.
 */
void merge_2_blocks(
    RecordInputStream& instream1,
    RecordInputStream& instream2,
    RecordOutputStream& output_stream,
    size_t record_size)
{
    // Input blocks should not be empty.
    assert(!instream1.empty());
    assert(!instream2.empty());

    const unsigned char * rec1 = instream1.record();
    const unsigned char * rec2 = instream2.record();

    // Merge until one stream runs empty.
    while (true) {

        // Choose which record should go first.
        if (record_compare(rec1, rec2, record_size) < 0) {
            // Push record from stream 1 and load next record.
            output_stream.put(rec1);
            instream1.next_record();
            if (instream1.empty()) {
                rec1 = NULL;
                break;
            }
            rec1 = instream1.record();
        } else {
            // Push record from stream 2 and load next record.
            output_stream.put(rec2);
            instream2.next_record();
            if (instream2.empty()) {
                rec2 = NULL;
                break;
            }
            rec2 = instream2.record();
        }
    }

    // At most one of the streams still has records left.
    // Copy those records to the output.

    while (!instream1.empty()) {
        output_stream.put(instream1.record());
        instream1.next_record();
    }

    while (!instream2.empty()) {
        output_stream.put(instream2.record());
        instream2.next_record();
    }
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
void merge_n_blocks(
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
 * @param ctx                   Reference to context structure.
 */
void merge_pass(
    BinaryFile& input_file,
    BinaryFile& output_file,
    uint64_t records_per_block,
    uint64_t num_blocks,
    unsigned int branch_factor,
    bool filter_dupl,
    const SortContext& ctx)
{
    assert(branch_factor > 1);
    assert(branch_factor <= num_blocks);

    // Only filter duplicates when the output is a single block.
    assert((!filter_dupl) || (branch_factor == num_blocks));

    Timer timer;
    timer.start();

    // Calculate number of buffers:
    // 2 buffers per input stream + more buffers for output stream.
    size_t num_output_buffers = 2 + (branch_factor - 1) / 2;
    size_t num_buffers = 2 * branch_factor + num_output_buffers;

    // Calculate buffer size.
    // Must be a multiple of the record size and the transfer alignment.
    size_t buffer_size = ctx.memory_size / num_buffers;
    buffer_size -= buffer_size % (TRANSFER_ALIGNMENT * ctx.record_size);

// TODO : double-buffering with I/O in separate thread

    // Initialize input streams.
    std::vector<std::unique_ptr<RecordInputStream>> input_streams;
    for (unsigned int i = 0; i < branch_factor; i++) {
        uint64_t block_size = records_per_block * ctx.record_size;
        uint64_t start_offset = i * block_size;
        uint64_t block_stride = branch_factor * block_size;
        if (start_offset >= input_file.size()) {
            break;
        }
        input_streams.emplace_back(new RecordInputStream(
            input_file,
            ctx.record_size,
            start_offset,
            block_size,
            block_stride,
            buffer_size));
    }

    // Initialize output stream.
    RecordOutputStream output_stream(
        output_file,
        ctx.record_size,
        0,
        buffer_size);

    // Loop over groups of blocks to be sorted.
    // Every group consists of "branch_factor" blocks, except the last
    // group which may contain fewer blocks.
    // Each group produces one output block.
    uint64_t block_index = 0;
    while (block_index < num_blocks) {

        // Determine how many blocks will be merged in this group.
        unsigned int this_branch_factor = branch_factor;
        if (branch_factor > num_blocks - block_index) {
            this_branch_factor = num_blocks - block_index;
        }

        // Skip to the next section of each active input stream.
        for (unsigned int i = 0; i < this_branch_factor; i++) {
            input_streams[i]->next_block();
        }

        if (this_branch_factor == 1) {

            // Last group contains just 1 block.
            // Copy it to the output.
            assert(!filter_dupl);
            RecordInputStream * instream = input_streams[0].get();
            while (!instream->empty()) {
                output_stream.put(instream->record());
                instream->next_record();
            }

        } else if (this_branch_factor == 2 && !filter_dupl) {

            // Special case for merging 2 blocks.
            merge_2_blocks(
                *input_streams[0],
                *input_streams[1],
                output_stream,
                ctx.record_size);

        } else {

            // Merge more than 2 blocks or filter duplicates.
            merge_n_blocks(
                input_streams,
                output_stream,
                ctx.record_size,
                this_branch_factor,
                filter_dupl);

        }

        // Skip to the start of the next block group.
        block_index += this_branch_factor;
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
    // Use blocks that are at most half of available memory,
    // so we can use two buffers to overlap I/O and sorting.
    uint64_t max_sort_block_size = ctx.memory_size / 2;

    // Calculate number of records per block.
    // Make sure this is a multiple of the transfer alignment size.
    uint64_t records_per_sort_block = max_sort_block_size / ctx.record_size;
    records_per_sort_block -= records_per_sort_block % TRANSFER_ALIGNMENT;

    // Calculate number of blocks during the initial sort pass.
    uint64_t num_records = file_size / ctx.record_size;
    uint64_t num_sort_blocks = 1 + (num_records - 1) / records_per_sort_block;

    SortStrategy strategy;
    strategy.records_per_sort_block = records_per_sort_block;
    strategy.num_sort_blocks = num_sort_blocks;

    // Plan the merge passes.
    // Start with the result of the initial sort pass.
    uint64_t records_per_block = records_per_sort_block;
    uint64_t num_blocks = num_sort_blocks;

    // Keep merging until there is only one block left.
    while (num_blocks > 1) {
        // Calculate the number of blocks out of this merge pass.
        uint64_t num_merged_blocks = 1 + (num_blocks - 1) / ctx.branch_factor;

        // Choose the smallest branch factor that produces this nr of blocks.
        unsigned int branch_factor = 1 + (num_blocks - 1) / num_merged_blocks;

        SortStrategy::MergePass merge_pass;
        merge_pass.records_per_input_block = records_per_block;
        merge_pass.num_input_blocks = num_blocks;
        merge_pass.branch_factor = branch_factor;
        strategy.merge_pass.push_back(merge_pass);

        // Result of this merge pass will go into the next pass.
        records_per_block *= branch_factor;
        num_blocks = num_merged_blocks;
    }

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

        // The merge passes alternate between tempfile-to-outputfile and
        // outputfile-to-tempfile.
        // The final merge pass will be tempfile-to-outputfile.
        // Depending on the number of merge passes, the initial sort pass
        // will either be inputfile-to-tempfile or inputfile-to-outputfile.
        BinaryFile * output_or_temp_file[2] = { &output_file, &temp_file };
        BinaryFile * sort_output_file =
            output_or_temp_file[num_merge_pass % 2];

        // Execute the initial sort pass.
        sort_pass(
            input_file,
            *sort_output_file,
            strategy.records_per_sort_block,
            strategy.num_sort_blocks,
            ctx);

        // Execute the merge passes.
        for (unsigned int mp = 0; mp < num_merge_pass; mp++) {
            log(ctx,
                "running merge pass %u / %u: "
                "%" PRIu64 " blocks, branch factor %u\n",
                mp,
                num_merge_pass,
                strategy.merge_pass[mp].num_input_blocks,
                strategy.merge_pass[mp].branch_factor);

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
                strategy.merge_pass[mp].records_per_input_block,
                strategy.merge_pass[mp].num_input_blocks,
                strategy.merge_pass[mp].branch_factor,
                filter_dupl,
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
        "  -s, --size=N    specify record size of N bytes (required)\n"
        "  -u, --unique    eliminate duplicates after sorting\n"
        "  --memory=M      use at most M MByte RAM (default: %d)\n"
        "  --branch=B      merge at most B arrays in one step (default: %d)\n"
        "  --temporary-directory=DIR  write temporary file to the specified\n"
        "                             directory (default: $TMPDIR)\n"
        "\n"
        "The output file must not yet exist.\n"
        "If the data does not fit in memory, a temporary file will be\n"
        "created with the same size as the input/output files.\n"
        "\n",
        DEFAULT_MEMORY_SIZE_MBYTE,
        DEFAULT_BRANCH_FACTOR);
}


} // anonymous namespace


int main(int argc, char **argv)
{
    const struct option longopts[] = {
        { "size", 1, NULL, 's' },
        { "unique", 0, NULL, 'u' },
        { "memory", 1, NULL, 'M' },
        { "branch", 1, NULL, 'B' },
        { "temporary-directory", 1, NULL, 'T' },
        { "verbose", 0, NULL, 'v' },
        { "help", 0, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };
    bool flag_unique = false;
    bool flag_verbose = false;
    int record_size = 0;
    int memory_size = DEFAULT_MEMORY_SIZE_MBYTE;
    int branch_factor = DEFAULT_BRANCH_FACTOR;
    std::string tempdir = get_default_tmpdir();
    int opt;

    while ((opt = getopt_long(argc, argv, "s:T:uv", longopts, NULL)) != -1) {
        switch (opt) {
            case 's':
                record_size = atoi(optarg);
                if (record_size < 1) {
                    fprintf(stderr,
                        "ERROR: Invalid record size (must be at least 1)\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'u':
                flag_unique = true;
                break;
            case 'M':
                memory_size = atoi(optarg);
                if (memory_size <= 0) {
                    fprintf(stderr, "ERROR: Invalid memory size\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'B':
                branch_factor = atoi(optarg);
                if (branch_factor < 2) {
                    fprintf(stderr,
                        "ERROR: Invalid radix value, must be at least 2\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'T':
                tempdir = optarg;
                break;
            case 'v':
                flag_verbose = true;
                break;
            case 'h':
                usage();
                return EXIT_SUCCESS;
            default:
                usage();
                return EXIT_FAILURE;
        }
    }

    if (record_size < 1) {
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

    if ((unsigned int)memory_size >= SIZE_MAX / 1024 / 1024) {
        fprintf(
            stderr,
            "ERROR: This system can allocate at most %zu MB memory\n",
            SIZE_MAX / 1024 / 1024 - 1);
        return EXIT_FAILURE;
    }

    std::string input_name(argv[optind]);
    std::string output_name(argv[optind+1]);

    SortContext ctx;
    ctx.record_size = record_size;
    ctx.memory_size = size_t(memory_size) * 1024 * 1024;
    ctx.branch_factor = branch_factor;
    ctx.flag_unique = flag_unique;
    ctx.flag_verbose = flag_verbose;
    ctx.temporary_directory = tempdir;

    try {
        sortbin(input_name, output_name, ctx);
    } catch (const std::exception& ex) {
        fprintf(stderr, "ERROR: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
