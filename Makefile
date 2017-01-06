# makefile for infdecoder

BIN = nnjm
BINDIR = ./bin
SRCDIR = ./source
TESTDIR = ./test

CXX = g++
CFLAGS = -Wall -fopenmp -O3 -funroll-loops -ffast-math -I ./include\
	 -I ./library -std=c++0x -DNDEBUG -DEIGEN_NO_DEBUG
LDFLAGS = -lgomp -lpthread -lm

SRC = $(wildcard $(SRCDIR)/*.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cpp
	echo compiling...$<
	$(CXX) $(CFLAGS) -c $< -o $@

# Link all Object Files with external Libraries into Binaries
$(BIN): dir $(OBJ)
	echo linking...
	$(CXX) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $(BINDIR)/$(BIN)

lib:
	ar -rv libnnjm.a $(SRCDIR)/constraint.o $(SRCDIR)/ffnn.o \
	$(SRCDIR)/io.o $(SRCDIR)/mathlib.o $(SRCDIR)/misc.o \
	$(SRCDIR)/model.o $(SRCDIR)/nnjm.o $(SRCDIR)/nnlm.o \
	$(SRCDIR)/symbol.o $(SRCDIR)/symtab.o $(SRCDIR)/utility.o \
	$(SRCDIR)/vocab.o $(SRCDIR)/parameter.o

dir:
	-mkdir -p $(BINDIR)

.PHONY: clean
clean:
	-rm -f $(SRCDIR)/*.o
	-rm -f -r $(BINDIR)
