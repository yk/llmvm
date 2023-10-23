CC = gcc
CFLAGS = -Wall -S -m32 -fverbose-asm -O3

all: hello helloasm

hello: hello.c
	$(CC) -o hello hello.c

helloasm: hello.c
	$(CC) $(CFLAGS) -o hello.asm hello.c

helloll: hello.c
	clang -S -emit-llvm hello.c

snek: snek.c
	$(CC) -o snek snek.c

snekasm: snek.c
	$(CC) $(CFLAGS) -o snek.asm snek.c

snekll: snek.c
	clang -S -emit-llvm -m32 snek.c

clean:
	rm -f hello snek hello.asm hello.ll snek.asm snek.ll

