CC = gcc
CFLAGS = -Wall -S -m32 -fverbose-asm -O3

all: fizzbuzzll snekll

fizzbuzz: fizzbuzz.c
	$(CC) -o fizzbuzz fizzbuzz.c

fizzbuzzasm: fizzbuzz.c
	$(CC) $(CFLAGS) -o fizzbuzz.asm fizzbuzz.c

fizzbuzzll: fizzbuzz.c
	clang -S -emit-llvm fizzbuzz.c

snek: snek.c
	$(CC) -o snek snek.c

snekasm: snek.c
	$(CC) $(CFLAGS) -o snek.asm snek.c

snekll: snek.c
	clang -S -emit-llvm -m32 snek.c

clean:
	rm -f hello snek hello.asm hello.ll snek.asm snek.ll

