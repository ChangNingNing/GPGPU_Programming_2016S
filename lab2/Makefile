INCLUDES=../utils
CXXFLAGS=-std=c++11 -O2 -arch sm_30

all: main

main: main.cu counting.cu
	nvcc -I$(INCLUDES) $(CXXFLAGS) main.cu counting.cu -o main

clean:
	rm main
