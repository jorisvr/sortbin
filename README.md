
# SortBin: a tool for sorting binary records

SortBin is a tool for sorting arrays of binary data records.
It is similar to the Unix `sort` utility.
But where `sort` works with lines of text, SortBin works with fixed-length binary data records.

SortBin reads input from a file, sorts it, and writes the sorted data to an output file.
These files contain flat, raw arrays of binary data records.

Records are interpreted as fixed-length strings of 8-bit unsigned integers.
These records are sorted in lexicographic order.
This means that records are sorted by their first byte, then records with equal first bytes are sorted by their second bytes, and so on.

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
build/sortbin --size=10 --memory=2G input.dat output.dat
```

* _size_ specifies the record size in bytes
* _memory_ specifies the amount of RAM that SortBin may use

