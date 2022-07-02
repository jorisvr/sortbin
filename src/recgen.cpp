/*
 * Tool to generate random binary data records.
 *
 * Written by Joris van Rantwijk in 2022.
 */

#define _FILE_OFFSET_BITS 64

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <vector>


#define MAX_RECORD_SIZE             65536
#define REPORT_INTERVAL             1000000


namespace {  // anonymous namespace


/**
 * Pseudo random number generator "xoroshiro128+"
 *
 * This code is based on the reference implementation,
 * modified by Joris van Rantwijk to fit in a C++ class.
 *
 * Source: http://prng.di.unimi.it/
 *
 * The following commpents apply to the reference implementation:
 *
 * Written in 2016-2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
 *
 * To the extent possible under law, the author has dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * See <http://creativecommons.org/publicdomain/zero/1.0/>.
 *
 * This is xoroshiro128+ 1.0, our best and fastest small-state generator
 * for floating-point numbers. We suggest to use its upper bits for
 * floating-point generation, as it is slightly faster than
 * xoroshiro128++/xoroshiro128**. It passes all tests we are aware of
 * except for the four lower bits, which might fail linearity tests (and
 * just those), so if low linear complexity is not considered an issue (as
 * it is usually the case) it can be used to generate 64-bit outputs, too;
 * moreover, this generator has a very mild Hamming-weight dependency
 * making our test (http://prng.di.unimi.it/hwd.php) fail after 5 TB of
 * output; we believe this slight bias cannot affect any application. If
 * you are concerned, use xoroshiro128++, xoroshiro128** or xoshiro256+.
 *
 * We suggest to use a sign test to extract a random Boolean value, and
 * right shifts to extract subsets of bits.
 *
 * The state must be seeded so that it is not everywhere zero. If you have
 * a 64-bit seed, we suggest to seed a splitmix64 generator and use its
 * output to fill s.
 *
 * NOTE: the parameters (a=24, b=16, b=37) of this version give slightly
 * better results in our test than the 2016 version (a=55, b=14, c=36).
 */
class Xoroshiro128plus
{
public:
    Xoroshiro128plus(uint64_t s0, uint64_t s1)
    {
        m_state[0] = s0;
        m_state[1] = s1;
    }

    inline uint64_t next()
    {
        const uint64_t s0 = m_state[0];
        uint64_t s1 = m_state[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        m_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        m_state[1] = rotl(s1, 37);

        return result;
    }

private:
    static inline uint64_t rotl(uint64_t x, int k)
    {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t m_state[2];
};



class RecordGenerator
{
public:
    RecordGenerator(
        unsigned int record_size,
        unsigned long long num_records,
        double duplicate_fraction,
        bool flag_ascii)
      : m_record_size(record_size),
        m_bits_per_record(0),
        m_highbit_threshold(0),
        m_salt(0),
        m_flag_ascii(flag_ascii),
        m_make_duplicates(false),
        m_salt_initialized(false)
    {
        assert(record_size > 0);

        if (duplicate_fraction > 0) {

            // Target a specific fraction of duplicate records.
            // We do this by defining a limited set of keys, such that each
            // key maps to a random record. To generate a record, we draw
            // uniformly from the set of keys, instead of from the set of
            // all possible records.

            // Calculate target number of unique records.
            double target_unique = num_records * (1 - duplicate_fraction);

            // Calculate the amount of information per record (in bits).
            double info_per_record =
                m_flag_ascii ?
                    ((record_size - 1) * log2(36.0))
                    : (record_size * 8.0);

            // Determine how many unique keys we need to draw to
            // get an exepected number of unique records that matches
            // our target.
            //
            // Draw N records from a set of V possible values.
            // Expected number of unique values:
            //
            //   U = V * (1 - (1 - 1/V)**N)
            //
            // Solve for N:
            //
            //   N = log(1 - U/V) / log(1 - 1/V)
            //
            if (info_per_record >= 128) {
                // The number of records is so big that we can just pretend
                // that every key will produce a different record.
            } else {
                double v = exp(info_per_record * M_LN2);

                if (target_unique * 1.000001 >= v) {
                    // There are not enough different records to produce
                    // the requested number of unique records.
                    // Just produce as many as possible.
                    target_unique = num_records;
                } else {

                    // log(1 - 1/V) is inaccurate for very large values of V;
                    // approximate as (-1/V)
                    double t =
                        (v < 1.0e6) ?
                            log(1.0 - 1.0 / v)
                            : (-1.0 / v);

                    // log(1 - U/V) is inaccurate for very large values of V;
                    // approximate as (-U/V)
                    double q =
                        (v < 1.0e6 * target_unique) ?
                            log(1.0 - target_unique / v)
                            : (- target_unique / v);

                    // Calculate the target number of unique keys.
                    target_unique = q / t;
                }
            }

            // Determine the number of random bits for which the
            // expected number of unique keys matches our target.
            // First scan in steps of 1 bit.
            unsigned int need_bits = 2;
            while (need_bits < 127) {
                double expected_unique =
                    expected_num_unique(num_records, need_bits);
                if (expected_unique >= target_unique) {
                    break;
                }
                need_bits++;
            }

            // Fine scan in steps of 1/16 bit.
            unsigned int need_bits_frac16 = 0;
            while (need_bits_frac16 < 16) {
                double nbits = need_bits - 1 + need_bits_frac16 / 16.0;
                double expected_unique =
                    expected_num_unique(num_records, nbits);
                if (expected_unique >= target_unique) {
                    break;
                }
                need_bits_frac16++;
            }

            if (need_bits < 127) {
                // Use this number of bits per record.
                m_bits_per_record = need_bits;
                m_highbit_threshold =
                    exp((63 + need_bits_frac16 / 16.0) * M_LN2);
                m_make_duplicates = true;
            } else {
                // We need so many random bits that nobody will notice
                // if we just use a uniform distribution of records.
                // So let's do that.
                m_make_duplicates = false;
            }
        }
    }

    void generate_record(unsigned char * record, Xoroshiro128plus& rng)
    {
        if (m_make_duplicates) {
            // We have a budget of fewer than 128 random bits per record.
            // Create a random seed value of that many bits.
            // Then use it to initialize a secondary random number generator.
            // Then use that generator to generate the actual record.

            // During the first call, generate a salt for the secondary
            // random number generator. Without this salt, every run
            // would sample from the same subset of records.
            if (!m_salt_initialized) {
                m_salt = rng.next();
                m_salt_initialized = true;
            }

            uint64_t s0 = 0, s1 = 0;
            unsigned int need_bits = m_bits_per_record;
            if (need_bits > 64) {
                s0 = rng.next();
                need_bits -= 64;
            }
            do {
                s1 = rng.next();
            } while (s1 > m_highbit_threshold);
            s1 >>= (64 - need_bits);
            s0 ^= m_salt;

            // Create secondary random number generator.
            Xoroshiro128plus rng2(s0, s1);
            rng2.next();
            rng2.next();

            // Use it to generate a record.
            generate_uniform_record(record, rng2);
        } else {
            // Uniform distribution of records.
            generate_uniform_record(record, rng);
        }
    }

private:
    void generate_uniform_record(unsigned char * record, Xoroshiro128plus& rng)
    {
        if (m_flag_ascii) {

            // Generate ASCII record.
            for (unsigned int i = 0; i < m_record_size - 1; i++) {
                uint64_t r = rng.next() >> 32;
                unsigned int p = r % 36;
                if (p < 10) {
                    record[i] = '0' + p;
                } else {
                    record[i] = 'a' + (p - 10);
                }
            }

            // Append newline.
            record[m_record_size-1] = '\n';

        } else {

            // Generate binary record.
            for (unsigned int i = 0; i < m_record_size; i++) {
                uint64_t r = rng.next() >> 32;
                record[i] = r & 0xff;
            }

        }
    }

    /** Calculate the expected number of unique records. */
    double expected_num_unique(unsigned long long num_records,
                               double bits_per_record)
    {
        // We draw N records from a set of V values with replacement.
        //
        // The expected number of unique values drawn in one batch is
        //
        //   V * (1 - (1 - 1/V)**N)
        //

        double v = exp(bits_per_record * M_LN2);

        // The calculation (1 - 1/V)**N is inaccurate for very large
        // values of V. In that case, approximation as exp(- N / V).
        double t =
            (v < 1.0e6) ?
                pow(1.0 - 1.0 / v, num_records)
                : exp(- double(num_records) / v);

        return v * (1.0 - t);
    }

    unsigned int m_record_size;
    unsigned int m_bits_per_record;
    uint64_t m_highbit_threshold;
    uint64_t m_salt;
    bool m_flag_ascii;
    bool m_make_duplicates;
    bool m_salt_initialized;
};


int recgen(
    const char *output_name,
    unsigned int record_size,
    unsigned long long num_records,
    double duplicate_fraction,
    bool flag_ascii,
    uint64_t seed)
{
    Xoroshiro128plus rng(seed, 0);
    rng.next();
    rng.next();

    RecordGenerator record_generator(
        record_size,
        num_records,
        duplicate_fraction,
        flag_ascii);

    int fd = open(output_name, O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Can not create output file (%s)\n",
                strerror(errno));
        return -1;
    }

    FILE *outf = fdopen(fd, "w");
    if (outf == nullptr) {
        fprintf(stderr, "ERROR: fdopen() failed (%s)\n", strerror(errno));
        close(fd);
        return -1;
    }

    std::vector<unsigned char> record(record_size);

    for (unsigned long long i = 0; i < num_records; i++) {
        if ((i % REPORT_INTERVAL) == 0) {
            printf("\rgenerated %llu / %llu records    ", i, num_records);
            fflush(stdout);
        }

        record_generator.generate_record(record.data(), rng);

        if (fwrite(record.data(), record_size, 1, outf) != 1) {
            fprintf(stderr, "ERROR: Writing to output file failed (%s)\n",
                    strerror(errno));
            fclose(outf);
            return -1;
        }
    }

    printf("\rgenerated %llu records - done        \n", num_records);
    fflush(stdout);

    fclose(outf);

    return 0;
}


void usage()
{
    fprintf(stderr,
        "\n"
        "Generate fixed-length random binary records.\n"
        "\n"
        "Usage: recgen [-a] [-d D] [-S R] -n N -s S outputfile\n"
        "\n"
        "Options:\n"
        "\n"
        "  -n N        specify number of records (required)\n"
        "  -s S        specify record size in bytes (required)\n"
        "  -a          generate ASCII records: 0-1, a-z, end in newline\n"
        "  -d D        specify fraction of duplicate records (0.0 to 1.0)\n"
        "  -S R        specify seed for random generator (default 1)\n"
        "\n");
}


} // anonymous namespace


int main(int argc, char **argv)
{
    double duplicate_fraction = 0.0;
    unsigned long long num_records = 0;
    unsigned long record_size = 0;
    bool flag_ascii = false;
    unsigned long long seed = 1;
    int opt;

    while ((opt = getopt(argc, argv, "ad:n:s:S:")) != -1) {
        char *endptr;
        switch (opt) {
            case 'a':
                flag_ascii = true;
                break;
            case 'd':
                duplicate_fraction = strtod(optarg, &endptr);
                if (endptr == optarg
                        || *endptr != '\0'
                        || duplicate_fraction < 0.0
                        || duplicate_fraction > 1.0) {
                    fprintf(stderr,
                        "ERROR: Invalid duplicate fraction"
                        " (must be between 0.0 and 1.0)\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'n':
                num_records = strtoull(optarg, &endptr, 10);
                if (endptr == optarg
                        || *endptr != '\0'
                        || num_records == 0) {
                    fprintf(stderr,
                        "ERROR: Invalid number of records\n");
                    return EXIT_FAILURE;
                }
                break;
            case 's':
                record_size = strtoul(optarg, &endptr, 10);
                if (endptr == optarg
                        || *endptr != '\0'
                        || record_size < 2
                        || record_size > MAX_RECORD_SIZE) {
                    fprintf(stderr,
                        "ERROR: Invalid record size\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'S':
                seed = strtoull(optarg, &endptr, 10);
                if (endptr == optarg
                        || *endptr != '\0'
                        || seed > UINT64_MAX) {
                    fprintf(stderr,
                        "ERROR: Invalid random seed\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'h':
                usage();
                return EXIT_SUCCESS;
            default:
                usage();
                return EXIT_FAILURE;
        }
    }

    if (num_records == 0) {
        fprintf(stderr, "ERROR: Missing required parameter -n\n");
        usage();
        return EXIT_FAILURE;
    }
    if (record_size == 0) {
        fprintf(stderr, "ERROR: Missing required parameter -s\n");
        usage();
        return EXIT_FAILURE;
    }

    if (argc < optind + 1) {
        fprintf(stderr,
                "ERROR: Output file name must be specified\n");
        usage();
        return EXIT_FAILURE;
    }

    if (argc > optind + 1) {
        fprintf(stderr, "ERROR: Unexpected command-line parameters\n");
        usage();
        return EXIT_FAILURE;
    }

    const char * output_name = argv[optind];

    int ret = recgen(
        output_name,
        record_size,
        num_records,
        duplicate_fraction,
        flag_ascii,
        seed);

    return (ret == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
