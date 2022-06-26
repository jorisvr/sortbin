#
# Makefile for sortbin utility.
#

CXX = g++
CXXFLAGS = -Wall -O2 -pthread
  # -fsanitize=address -fsanitize=undefined

SRCDIR = src
BUILDDIR = build

TOOLS = sortbin recgen
BINFILES = $(patsubst %,$(BUILDDIR)/%,$(TOOLS))

.PHONY: all
all: $(BINFILES)

$(BUILDDIR)/sortbin: $(SRCDIR)/sortbin.cpp
$(BUILDDIR)/recgen: $(SRCDIR)/recgen.cpp

$(BUILDDIR)/%: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

.PHONY: clean
clean:
	$(RM) $(BINFILES)

