CC = gcc
FLAGS = -lm


all: v0

v0: v0.c utilities.c
	$(CC) -o $@ $^ $(FLAGS)

test: test.c utilities.c
	$(CC) -o $@ $^ $(FLAGS)
