INCLUDES=../utils
CXXFLAGS=-std=c++11 -O2 -arch sm_30

all: main

main: lab3.cu
	nvcc -I$(INCLUDES) $(CXXFLAGS) main.cu lab3.cu pgm.cpp -o main

clean:
	rm main
