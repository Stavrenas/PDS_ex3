CC = gcc
NVCC = nvcc
FLAGS = -lm

all: cpu

cpu: cpu.c utilities.c
	$(CC) -o $@ $^ $(FLAGS)

test: test.c utilities.c
	$(CC) -o $@ $^ $(FLAGS)

gpu: gpu.cu cudaUtilities.cu utilities.c
	$(NVCC) $^ -o $@  
