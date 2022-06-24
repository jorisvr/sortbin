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
#define REPORT_INTERVAL             100000


namespace {  // anonymous namespace


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
        m_flag_ascii(flag_ascii)
    {
        assert(record_size > 0);

        if (duplicate_fraction > 0) {
            // Calculate the number of bits needed to achieve the
            // specified duplicate fraction.
            //
            // This calculation is not exactly right, but it
            // gives reasonable results for duplication_fraction > 0.5.
            double num_values =
                double(num_records) * (1 / duplicate_fraction - 1);
            m_bits_per_record = lrint(ceil(log2(num_values)));
        } else {
            // Use uniform distribution of records.
            m_bits_per_record = 8 * record_size;
        }
    }

    void generate_record(unsigned char * record, Xoroshiro128plus& rng)
    {
        if (m_bits_per_record >= 8 * m_record_size || m_bits_per_record >= 128) {
            // Just generate uniformly selected records.
            // Nobody will notice the difference.
            generate_uniform_record(record, rng);
        } else {
            // We have a budget of fewer than 128 random bits per record.
            // Create a random seed value of that many bits, then use it
            // to initialize a secondary random number generator to generate
            // the data.
            uint64_t s0 = 0, s1 = 0;
            if (m_bits_per_record > 64) {
                s0 = rng.next();
                s1 = rng.next() >> (128 - m_bits_per_record);
            } else {
                s0 = rng.next() >> (64 - m_bits_per_record);
                s1 = 0;
            }
            Xoroshiro128plus rng2(s0, s1);
            rng2.next();
            rng2.next();
            generate_uniform_record(record, rng2);
        }
    }

private:
    void generate_uniform_record(unsigned char * record, Xoroshiro128plus& rng)
    {
        if (m_flag_ascii) {

            // Generate ASCII record.
            for (unsigned int i = 0; i < m_record_size - 1; i++) {
                uint64_t r = rng.next() >> 4;
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
                uint64_t r = rng.next() >> 4;
                record[i] = r & 0xff;
            }

        }
    }

    unsigned int m_record_size;
    unsigned int m_bits_per_record;
    bool m_flag_ascii;
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
    if (outf == NULL) {
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
        "Usage: recgen [-a] [-d D] -n N -s S outputfile\n"
        "\n"
        "Options:\n"
        "\n"
        "  -a          generate ASCII records: 0-1, a-z, end in newline\n"
        "  -d D        specify fraction of duplicate records (0.0 to 1.0)\n"
        "  -n N        specify number of records (required)\n"
        "  -s S        specify record size in bytes (required)\n"
        "\n");
}


} // anonymous namespace


int main(int argc, char **argv)
{
    double duplicate_fraction = 0.0;
    unsigned long long num_records = 0;
    unsigned long record_size = 0;
    bool flag_ascii = false;
    uint64_t seed = 1;
    int opt;

    while ((opt = getopt(argc, argv, "ad:n:s:")) != -1) {
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
