all: myexp

logger.o: logger.cpp logger.hpp CS149intrin.hpp CS149intrin.cpp
	g++ -c logger.cpp

CS149intrin.o: CS149intrin.cpp CS149intrin.hpp logger.cpp logger.hpp
	g++ -c CS149intrin.cpp

myexp: CS149intrin.o logger.o main.cpp
	g++ -I../common logger.o CS149intrin.o main.cpp -o myexp

clean:
	rm -f *.o myexp *~
