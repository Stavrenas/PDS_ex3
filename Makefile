CC = gcc
NVCC = nvcc
CFLAGS = -lm -O3

all: cpu

cpu: cpu.c utilities.c
	$(CC) -o $@ $^ $(CFLAGS)

test: test.c utilities.c
	$(CC) -o $@ $^ $(CFLAGS)

gpu: gpu.cu cudaUtilities.cu utilities.c
	$(NVCC) $^ -o $@  
