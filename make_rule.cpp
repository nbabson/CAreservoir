
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <string>
#include <iomanip>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
using namespace std;

int RULELENGTH = 64;
int STATES = 4;

void random_rule(vector<int>& );

int main() {
    ofstream out;

    out.open("random_4_state.txt", ofstream::out);
    vector<int> rule(64);
    srand(time(NULL));
    random_rule(rule);
    for (int j = 0; j < RULELENGTH; ++j)
       out << rule[j];
    out.close();
}

// Generate a random rule
void random_rule(vector<int>& rule) {
    for (int i = 0; i < RULELENGTH; ++i) 
	rule[i] = rand() % STATES;
}


