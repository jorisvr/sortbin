
# SortBin: a tool for sorting binary records

SortBin is a tool for sorting arrays of binary data records.
It is similar to the Unix `sort` utility.
But where `sort` works with lines of text, SortBin works with fixed-length binary data records.

SortBin reads input from a file, sorts it, and writes the sorted data to an output file.
These files contain flat, raw arrays of binary data records.

Records are interpreted as fixed-length strings of 8-bit unsigned integers.
These records are sorted in lexicographic order.
This means that records are sorted by their first byte, records with equal first bytes are sorted by their second bytes, and so on.

SortBin can sort very large data files which do not fit in memory.
In such cases, a temporary file is used to store intermediate results.
The program starts by separately sorting blocks of data that do fit in memory.
It then iteratively merges these blocks into larger sorted blocks until the complete file is sorted.

This program is designed to work with relatively short data records, up to about 20 bytes.
Sorting larger records should work, but may be inefficient.


## Usage

SortBin has only been tested on Linux.

To use SortBin, you must compile the source code.
Clone the repository, then build as follows:
```
git clone https://github.com/jorisvr/sortbin.git
cd sortbin
make
```

You can now sort data like this:
```
build/sortbin --size=10 input.dat output.dat
```
where _size_ specifies the record size in bytes.

The tool has a bunch of options that can be used to tune its performance:
```
Usage: sortbin [options] inputfile outputfile

Options:

  -s, --size=N
        Specify record size of N bytes (required)

  -u, --unique
        Eliminate duplicates after sorting

  --memory=<n>M, --memory=<n>G
        Specify the amount of RAM that may be used.
        Use suffix 'M' for MiByte, or 'G' for GiByte. (default: 1024 MiB)

  --branch=N
        Merge at most N subarrays in one step. (default: 16)

  --parallel=N
        Use N threads for parallel sorting. (default: 1)

  --iothread / --no-iothread
        Enable or disable use of background threads for I/O.
        (default: enable)

  -T, --temporary-directory=DIR
        Write temporary file to the specified directory. (default: $TMPDIR)

  -v, --verbose
        Write progress messages to STDERR.
```

For input files that fit in memory, the best way to get good performance is making sure that the tool can allocate enough memory to contain the file, and optionally increasing the number of threads.

For input files that do not fit in memory, the best way to get good performance is minimizing the number of required merge passes. This can be done by increasing the _branch factor_, and to a lesser degree by increasing the amount of memory.


## Status of this software

SortBin is just a hobby project, not production-quality software.

My limited testing suggests that the tool works correctly and is reasonably efficient.
Testing was done on Linux, but the tool should probably also work on other POSIX systems.

If you notice any bugs in SortBin, please let me know by making an issue in the Github repository.

I will try to respond to issue reports, but I can not promise to fix all bugs.
I will probably not work on feature requests or accept pull requests either.
This is, after all, just a hobby project.


## How it works

If the input file fits in memory, SortBin simply runs an in-place quicksort on the whole file.
Optionally, multiple threads may be used to run the quicksort algorithm on parallel CPU cores.

If the file does not fit in memory, SortBin uses a sort-then-merge strategy:
 - Create a temporary file with the same size as the input.
 - Chop the input into blocks that fit in memory. Read each block, sort it with quicksort, then write to the temporary file.
 - Read multiple sorted blocks from the temporary file and merge them into a bigger sorted block. By default, SortBin does a 16-way merge but this is configurable.
 - Continue merging sorted blocks into bigger sorted blocks until only 1 block is left. That block contains the final sorted output.

Optionally, multiple threads may be used to speed up the sorting phase by running the quicksort algorithm on parallel CPU cores.

By default, the sort-then-merge strategy delegates disk I/O to a separate thread so that I/O and CPU processing can occur simultaneously.


## Performance

I compared the performance of SortBin to that of GNU sort.
This is in many ways an unfair comparison.
The GNU sort utility was designed to sort lines of text, is highly customizable and is expected to show reasonable performance on a wide range of computer systems.
In contrast, SortBin can only sort fixed-length records, is not customizable, and was optimized for sorting big files of small records on a modern PC.

I tested the performance by sorting a 100 GB file using only 1 GB memory.
For GNU sort, I allowed 2 GB memory because it seems to use a significant amount of memory to keep track of lines of text.
For SortBin, I set the number of parallel sort threads to 4.

I tested on Linux, x86\_64 CPU with 6 cores, 3600 MHz, 5980 rpm SATA harddisk.
My performance figures should be taken with a big pinch of salt.
I have not checked how accurate these results reproduce on the same system or on other systems.
And I did not try very hard to keep background processes from potentially influencing the results.

| Input size | Records | Record size | Program  | Duration  | CPU time  |
| ---------- | ------- | ----------- | -------- | --------- | --------  |
| 100 GB     | 1e10    | 10 bytes    | GNU sort | 6.5 hours | 9.9 hours |
| 100 GB     | 1e10    | 10 bytes    | SortBin  | 1.2 hours | 1.0 hours |
| 100 GB     | 2e9     | 50 bytes    | GNU sort | 1.8 hours | 1.9 hours |
| 100 GB     | 2e9     | 50 bytes    | SortBin  | 1.2 hours | 0.4 hours |

For what it's worth, SortBin appears to be much faster than GNU sort when sorting small binary records.
This performance advantage decreases as records get bigger.


## Credits

While writing SortBin, I used ideas from several other software packages:
 - GNU sort, the standard `sort` utility of the GNU project, <br>
   part of coreutils: https://www.gnu.org/software/coreutils/
 - The implementation of `qsort()` in the GNU C Library. <br>
   https://www.gnu.org/software/libc/
 - The implementation of `std::sort()` in the GNU C++ Library. <br>
   https://gcc.gnu.org/onlinedocs/libstdc++/

To test SortBin, I wrote a tool to generate random binary records.
It uses the pseudo-random number generator Xoroshiro128+ from
https://prng.di.unimi.it/xoroshiro128plus.c


## License

Copyright (C) 2022 Joris van Rantwijk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/ .

