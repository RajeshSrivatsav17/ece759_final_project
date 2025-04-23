# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17 -fopenmp

# Project structure
SRC_DIR = .
TARGET = smoke

# Source files and target
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Default target
all: $(TARGET)

# Build the target directly from source files
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf smoke 

.PHONY: all clean