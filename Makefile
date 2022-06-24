#
# Makefile for sortbin utility.
#

CXX = g++
CXXFLAGS = -Wall -O2
  # -fsanitize=address -fsanitize=undefined

.PHONY: all
all: sortbin recgen

sortbin: sortbin.cpp
recgen: recgen.cpp

.PHONY: clean
clean:
	$(RM) sortbin recgen
