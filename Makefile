CA:
	g++ CAreservoir.cpp -O3 -c -o CAreservoir.o
	g++ CAreservoir.o -O3 dataanalysis.o ap.o alglibinternal.o alglibmisc.o linalg.o statistics.o specialfunctions.o solvers.o optimization.o -o CAreservoir


clean:
	rm -f CAreservoir CAreservoir.o
