all: clean build

clean:
	rm -f trace
	rm -f test.png

build:
	clang trace.c -o trace -lOpenCL -lpng -DCL_TARGET_OPENCL_VERSION=120
