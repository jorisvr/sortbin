#
# Makefile for sortbin utility.
#

CXX = g++
CXXFLAGS = -Wall -O2 -pthread
CXXFLAGS_DEBUG = -g -fsanitize=address -fsanitize=undefined -fsanitize=leak

SRCDIR = src
BUILDDIR = build

TOOLS = sortbin recgen
BINFILES = $(patsubst %,$(BUILDDIR)/%,$(TOOLS))
BINFILES_DEBUG = $(patsubst %,$(BUILDDIR)/%_dbg,$(TOOLS))

.PHONY: all
all: $(BINFILES)

.PHONY: test
test: $(BINFILES_DEBUG)
	cd tests ; ./run_tests.sh

$(BUILDDIR)/sortbin: $(SRCDIR)/sortbin.cpp
$(BUILDDIR)/recgen: $(SRCDIR)/recgen.cpp $(SRCDIR)/xoroshiro128plus.h

$(BUILDDIR)/sortbin_dbg: $(SRCDIR)/sortbin.cpp
$(BUILDDIR)/recgen_dbg: $(SRCDIR)/recgen.cpp $(SRCDIR)/xoroshiro128plus.h

$(BUILDDIR)/%: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

$(BUILDDIR)/%_dbg: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_DEBUG) $(LDFLAGS) $< $(LDLIBS) -o $@

.PHONY: clean
clean:
	$(RM) $(BINFILES) $(BINFILES_DEBUG)
	$(RM) -r tests/testdata

