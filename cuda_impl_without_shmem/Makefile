# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -arch=sm_70 -lineinfo

# Project structure
SRC_DIR = .
TARGET = smoke

# Source files and target
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Default target
all: $(TARGET)

# Build the target directly from source files
$(TARGET): $(SRCS)
    $(NVCC) $(CXXFLAGS) -o $@ $^

clean:
    rm -rf smoke 

.PHONY: all clean