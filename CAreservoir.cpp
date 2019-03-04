#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <time.h>
#include "dataanalysis.h"
#include "CAreservoir.h"

using namespace alglib;
using namespace std;

int main(int argc, char **argv) {

    CA ca;
    srand(time(NULL));

    cout << "in Main\n";

    return 0;
}


CA::CA() {
    bool unique;
    _map.resize(R, vector<int>(INPUT_LENGTH));
    _cell.resize(WIDTH, vector<int>(I));
    for (int i = 0; i < R; ++i) {
	for (int j = 0; j < INPUT_LENGTH; ++j) {
	    do {
		unique = true;
                _map[i][j] = rand() % DIFFUSE_LENGTH;
                for (int k = 0; k < j; ++k)
		    if (_map[i][j] == _map[i][k])
		      	unique = false;
	    } while (!unique);
	}
    }
}



