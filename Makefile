#
# Makefile for sortbin utility.
#

CXX = g++
CXXFLAGS = -Wall -O2
  # -fsanitize=address -fsanitize=undefined

sortbin: sortbin.cpp

.PHONY: clean
clean:
	$(RM) sortbin
