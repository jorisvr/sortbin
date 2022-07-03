#!/bin/bash

#
# This script runs a series of tests on the SortBin tool.
#
# The tests are relatively small.
# The whole test suite requires less than 1 GB disk space and 1 GB memory.
#
# This script assumes it runs from the "tests" subdirectory of the repository
# and expects the SortBin tools already built in the "build" subdirectory.
#
# This script writes temporary data files to a subdirectory "testdata"
# in the current working directory.
#

SCRIPTDIR=$(dirname "$0")

# Tools under test.
RECGEN="${SCRIPTDIR}/../build/recgen_dbg"
SORTBIN="${SCRIPTDIR}/../build/sortbin_dbg"

# GNU sort command.
SORT="sort"

# Directory for temporary test data.
TESTDATA="testdata"


# Run the "recgen" tool and show invocation.
recgen () {
    echo "${RECGEN} $*"
    "${RECGEN}" "$@"
}


# Run the Unix "sort" command and show invocation.
unixsort () {
    echo "${SORT} --temporary-directory=${TESTDATA} $*"
    "${SORT}" "--temporary-directory=${TESTDATA}" "$@"
}


# Run the "sortbin" tool and show invocation.
sortbin () {
    echo "${SORTBIN} --temporary-directory=${TESTDATA} --verbose $*"
    "${SORTBIN}" "--temporary-directory=${TESTDATA}" --verbose "$@"
}


# Verify SHA-256 checksum.
#   $1 = file name
#   $2 = expected SHA-256 sum
verify_sha256 () {
    ( cd "${TESTDATA}" ; echo "${2}  ${1}" | sha256sum -c )
}


# Count number of lines in output file.
#   $1 = output file name
count_unique () {
    local nline
    nline=$(wc --lines < "${TESTDATA}/$1")
    echo "$1: ${nline} unique records"
}


# Stop on the first error.
set -e

echo "Running tests"
echo

# Wipe and create test data directory, if needed.
[ ! -d "${TESTDATA}" ] || rm -r "${TESTDATA}"
mkdir -p "${TESTDATA}"


#
# Generate test input files
#

echo "Generating in01: 10 MB, 10 bytes/record, ascii, 50% duplicates"

hash_in01="fe72553a046d9e268c8c9da62902f63f1ba4b02901cd2413f9f1df7792e26304"
hash_out01="59df36e7f7895b56e59ef94cbad269cd5bec908d8a8e32aa2ca5ea624481431f"
hash_out01u="e46e33fbf34ca575652806a51d2042b64da073d76ed52e5e74c4becb85db974f"

recgen -S 1001 -s 10 -n 1000000 -a -d 0.5 "${TESTDATA}/in01"
verify_sha256 in01 ${hash_in01}
echo

echo "Generating in02: 10 MB, 10 bytes/record, ascii, 95% duplicates"

hash_in02="c5cdd3faeaf11b1a5508e3f781148a8b899514999f61917c4897bef8dc65aacf"
hash_out02u="a95f80b2400a1e14645f1d186a065a7483c103229fc94f11bb0d62e4da6edd7d"

recgen -S 1002 -s 10 -n 1000000 -a -d 0.95 "${TESTDATA}/in02"
verify_sha256 in02 ${hash_in02}
echo

echo "Generating in03: 100 MB, 10 bytes/record, ascii, 50% duplicates"

hash_in03="1a3c00a317f39e9dc336c2a46d31a3e3189a41285723f32a09be316d5273580c"
hash_out03="f17a2e566bbccecdab04373f92ba87784af77cb36c510fb7949d7fd719528a01"
hash_out03u="cbb975d1ce136e00668f14feb5880ebce474366ac5cb5c74287b7b4018c62b03"

recgen -S 1003 -s 10 -n 10000000 -a -d 0.5 "${TESTDATA}/in03"
verify_sha256 in03 ${hash_in03}
echo

echo "Generating in04: 100 MB, 100 bytes/record, ascii, 50% duplicates"

hash_in04="75fee029fbaa762176ec4f72ecf356674dda04cefebfb01af48b17ee560f9c25"
hash_out04u="4df05a28cdf2af29c4e8c5bb22930ee68bea12cca362ae378ab5dc6c6049fa9f"

recgen -S 1004 -s 100 -n 1000000 -a -d 0.5 "${TESTDATA}/in04"
verify_sha256 in04 ${hash_in04}
echo

echo "Generating in05: 10 MB, 10 bytes/record, binary, 50% duplicates"

hash_in05="aacd9d9b1955a88397004a8ca6192b20fb9bcfa60633ac735cb6297a997f38ad"
hash_out05u="f8b5da29076b36f7a97c9fc15da6a1754e4cc85aead368bf0540b9940f8f8cdb"

recgen -S 1005 -s 10 -n 1310720 -d 0.5 "${TESTDATA}/in05"
verify_sha256 in05 ${hash_in05}
echo


#
# Run GNU sort to check reference output.
#

echo "Running GNU sort to check reference output"
echo

unixsort -o "${TESTDATA}/out01" "${TESTDATA}/in01"
verify_sha256 out01 ${hash_out01}

unixsort -o "${TESTDATA}/out01u" --unique "${TESTDATA}/in01"
verify_sha256 out01u ${hash_out01u}
count_unique out01u

unixsort -o "${TESTDATA}/out02u" --unique "${TESTDATA}/in02"
verify_sha256 out02u ${hash_out02u}
count_unique out02u

unixsort -o "${TESTDATA}/out03" "${TESTDATA}/in03"
verify_sha256 out03 ${hash_out03}

unixsort -o "${TESTDATA}/out03u" --unique "${TESTDATA}/in03"
verify_sha256 out03u ${hash_out03u}
count_unique out03u

unixsort -o "${TESTDATA}/out04u" --unique "${TESTDATA}/in04"
verify_sha256 out04u ${hash_out04u}
count_unique out04u

echo


#
# Test in-memory sorting.
#

echo "in01: 10 MB, in-memory sort, non-parallel, without background I/O"
sortbin --size=10 --memory=100M --parallel=1 --no-iothread "${TESTDATA}/in01" "${TESTDATA}/out01_x"
verify_sha256 out01_x ${hash_out01}
rm "${TESTDATA}/out01_x"
echo

echo "in01: 10 MB, in-memory sort, non-parallel"
sortbin --size=10 --memory=100M --parallel=1 --iothread "${TESTDATA}/in01" "${TESTDATA}/out01_x"
verify_sha256 out01_x ${hash_out01}
rm "${TESTDATA}/out01_x"
echo

echo "in01: 10 MB, in-memory sort, parallel"
sortbin --size=10 --memory=100M --parallel=4 --iothread "${TESTDATA}/in01" "${TESTDATA}/out01_x"
verify_sha256 out01_x ${hash_out01}
rm "${TESTDATA}/out01_x"
echo

echo "in01: 10 MB, in-memory sort, unique, parallel"
sortbin --size=10 --memory=100M --parallel=4 --iothread --unique "${TESTDATA}/in01" "${TESTDATA}/out01u_x"
verify_sha256 out01u_x ${hash_out01u}
rm "${TESTDATA}/out01u_x"
echo

echo "in02: 10 MB, many duplicates, in-memory sort, unique, parallel"
sortbin --size=10 --memory=100M --parallel=4 --iothread --unique "${TESTDATA}/in02" "${TESTDATA}/out02u_x"
verify_sha256 out02u_x ${hash_out02u}
rm "${TESTDATA}/out02u_x"
echo

echo "in03: 100 MB, in-memory sort, unique, parallel"
sortbin --size=10 --memory=100M --parallel=4 --iothread --unique "${TESTDATA}/in03" "${TESTDATA}/out03u_x"
verify_sha256 out03u_x ${hash_out03u}
rm "${TESTDATA}/out03u_x"
echo

echo "in04: 100 MB, big records, in-memory sort, unique, parallel"
sortbin --size=100 --memory=100M --parallel=4 --iothread --unique "${TESTDATA}/in04" "${TESTDATA}/out04u_x"
verify_sha256 out04u_x ${hash_out04u}
rm "${TESTDATA}/out04u_x"
echo

echo "in05: 10 MB, binary data, in-memory sort, unique, parallel"
sortbin --size=10 --memory=100M --parallel=4 --iothread --unique "${TESTDATA}/in05" "${TESTDATA}/out05u_x"
verify_sha256 out05u_x ${hash_out05u}
rm "${TESTDATA}/out05u_x"
echo


#
# Test external sorting.
#

echo "in03: 100 MB, external sort, 1 merge pass"
sortbin --size=10 --memory=20M --parallel=4 --iothread "${TESTDATA}/in03" "${TESTDATA}/out03_x"
verify_sha256 out03_x ${hash_out03}
rm "${TESTDATA}/out03_x"
echo

echo "in03: 100 MB, external sort, 1 merge pass, unique"
sortbin --size=10 --memory=20M --parallel=4 --iothread --unique "${TESTDATA}/in03" "${TESTDATA}/out03u_x"
verify_sha256 out03u_x ${hash_out03u}
rm "${TESTDATA}/out03u_x"
echo

echo "in05: 10 MB, binary data, external sort, 1 merge pass, equal block sizes"
sortbin --size=10 --memory=5M --parallel=4 --iothread --unique "${TESTDATA}/in05" "${TESTDATA}/out05u_x"
verify_sha256 out05u_x ${hash_out05u}
rm "${TESTDATA}/out05u_x"
echo

echo "in03: 100 MB, external sort, 2 merge passes, unique, without background I/O"
sortbin --size=10 --memory=10M --parallel=4 --no-iothread --branch=4 --unique "${TESTDATA}/in03" "${TESTDATA}/out03u_x"
verify_sha256 out03u_x ${hash_out03u}
rm "${TESTDATA}/out03u_x"
echo

echo "in03: 100 MB, external sort, 2 merge passes, unique"
sortbin --size=10 --memory=20M --parallel=4 --iothread --branch=4 --unique "${TESTDATA}/in03" "${TESTDATA}/out03u_x"
verify_sha256 out03u_x ${hash_out03u}
rm "${TESTDATA}/out03u_x"
echo

echo "in03: 100 MB, external sort, 2 merge passes, balanced merge tree"
sortbin --size=10 --memory=22M --parallel=4 --iothread --branch=3 "${TESTDATA}/in03" "${TESTDATA}/out03_x"
verify_sha256 out03_x ${hash_out03}
rm "${TESTDATA}/out03_x"
echo

echo "in03: 100 MB, external sort, 2 merge passes, slightly unbalanced merge"
sortbin --size=10 --memory=20M --parallel=4 --iothread --branch=9 "${TESTDATA}/in03" "${TESTDATA}/out03_x"
verify_sha256 out03_x ${hash_out03}
rm "${TESTDATA}/out03_x"
echo

echo "in03: 100 MB, external sort, 2 merge passes, slightly unbalanced merge"
sortbin --size=10 --memory=13M --parallel=4 --iothread --branch=4 "${TESTDATA}/in03" "${TESTDATA}/out03_x"
verify_sha256 out03_x ${hash_out03}
rm "${TESTDATA}/out03_x"
echo

echo "in03: 100 MB, external sort, 2-way merging, 5 merge passes"
sortbin --size=10 --memory=10M --parallel=4 --iothread --branch=2 "${TESTDATA}/in03" "${TESTDATA}/out03_x"
verify_sha256 out03_x ${hash_out03}
rm "${TESTDATA}/out03_x"
echo

echo "in03: 100 MB, external sort, 24-way merge, 1 merge pass"
sortbin --size=10 --memory=8M --parallel=4 --iothread --branch=24 "${TESTDATA}/in03" "${TESTDATA}/out03_x"
verify_sha256 out03_x ${hash_out03}
rm "${TESTDATA}/out03_x"
echo

echo "Tests finished"
