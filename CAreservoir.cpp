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
    real_2d_array training_data;
    // Add one for target

    srand(time(NULL));
    ca.set_rule(RULE90);
    training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);

    cout << "in Main\n";

    return 0;
}



/***************************************************************************************/

CA::CA() {
    bool unique;
    int i, j, k;

    _iter = 0;
    _map.resize(R, vector<int>(INPUT_LENGTH));
    _cell.resize(WIDTH, vector<int>(I));
    _rule.resize(RULELENGTH);
    // Initialize first row with 0s
    for (i = 0; i < WIDTH; ++i)
	_cell[i][0] = 0;
    for (i = 0; i < R; ++i) {
	for (j = 0; j < INPUT_LENGTH; ++j) {
	    do {
		unique = true;
                _map[i][j] = rand() % DIFFUSE_LENGTH;
                for (k = 0; k < j; ++k)
		    if (_map[i][j] == _map[i][k])
		      	unique = false;
	    } while (!unique);
	}
    }
}


/***************************************************************************************/

void CA::set_input(std::vector<int> input) {
    int i, j;

    _iter = 0;
    for (i = 0; i < R; ++i) {
	for (j = 0; j < INPUT_LENGTH; ++j) {
	    // Overwrite initial row with mapped inputs
            _cell[0][i * DIFFUSE_LENGTH + _map[i][j]] = input[j];
	}
    }
}

/***************************************************************************************/

void CA::set_rule(std::vector<int> rule) {
    int i;

    for (i = 0; i < RULELENGTH; ++i)
	_rule[i] = rule[i];
}








