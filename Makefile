CAreservoir:
	g++ -g CAreservoir.cpp -c -o CAreservoir.o
	g++ -g CAreservoir.o dataanalysis.o ap.o alglibinternal.o alglibmisc.o linalg.o statistics.o specialfunctions.o solvers.o optimization.o -o CAreservoir


