CPP = g++
CPPFLAGS = -g -O4 -Wall -std=c++11 -fopenmp

TARGETS = CAreservoir CA

all: $(TARGETS)

CAreservoir: CAreservoir.o
	$(CPP) $(CPPFLAGS) -o CAreservoir CAreservoir.o -lalglib

CA: CA.o
	$(CPP) $(CPPFLAGS) -o CA CA.o

clean:
	-rm -f *.o *.dat SVM_model* SVM_results* \
	*.csv sk_density* ev_* $(TARGETS)
