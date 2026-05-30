CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -lm
SRC = src/embedding.c src/main.c
OUT = grand_pattern

all: $(OUT)

$(OUT): $(SRC) src/embedding.h
	$(CC) $(CFLAGS) -o $(OUT) $(SRC) -lm

test: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)

.PHONY: all test clean
