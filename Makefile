CA1:
	g++ CAreservoir.cpp -std=c++11 -O3  -fopenmp -c -o CAreservoir.o -g
	g++ CAreservoir.o -std=c++11 -O3  -fopenmp alglib_func.a -o CAreservoir -g
	g++ -std=c++11 -o CA CA.cpp

clean:
	rm -f CAreservoir.o *.dat SVM_model* SVM_results* *.csv sk_density* ev_*
