CC=gcc
CFLAGS=-Wall -Wextra -std=c99 -O2 -lm

all: test

test: test/test_all.c src/*.c
	$(CC) $(CFLAGS) -I include -o test_all test/test_all.c src/*.c -lm && ./test_all

clean:
	rm -f test_all

.PHONY: all test clean
