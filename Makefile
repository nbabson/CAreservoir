CA:
	g++ -Wall -g CAreservoir.cpp -c -o CAreservoir.o
	g++ -Wall -g CAreservoir.o dataanalysis.o ap.o alglibinternal.o alglibmisc.o linalg.o statistics.o specialfunctions.o solvers.o optimization.o -o CAreservoir


clean:
	rm -f CAreservoir CAreservoir.o
