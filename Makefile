all: clean build

clean:
	rm -f trace
	rm -f test.png

build:
	gcc trace.c -o trace -lOpenCL -lpng
