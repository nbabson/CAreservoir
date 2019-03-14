CA:
	g++ CAreservoir.cpp -O3 -fopenmp -c -o CAreservoir.o
	g++ CAreservoir.o -O3 -fopenmp dataanalysis.o ap.o alglibinternal.o alglibmisc.o linalg.o statistics.o specialfunctions.o solvers.o optimization.o -o CAreservoir


clean:
	rm -f CAreservoir.o *.dat SVM_model* SVM_results*
