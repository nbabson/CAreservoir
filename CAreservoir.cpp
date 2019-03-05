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

    cout << "Mapping input to CA reservoir\n";
    CA ca;
    real_2d_array training_data;
    vector<linearmodel> output(3);

    srand(time(NULL));
    ca.set_rule(RULE90);
    // Add one for target
    training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);

    cout << "Building training data\n";
    ca.train_5_bit(training_data);
    cout << "Building regression models\n";
    ca.build_5_bit_model(training_data, output);
    // for 32 iputs
    // evolve CA 210 times
    // build training data
    // affix targets and build models

    // test
    cout << "in Main      " << (-5) % 4 << "       \n";

    return 0;
}



/***************************************************************************************/

CA::CA() {
    bool unique;
    int i, j, k;

    _iter = 0;
    _map.resize(R, vector<int>(INPUT_LENGTH));
    _cell.resize(I + 1, vector<int>(WIDTH));
    _rule.resize(RULELENGTH);
    // Initialize first row with 0s
    for (i = 0; i < WIDTH; ++i)
	_cell[0][i] = 0;
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

void CA::set_input(vector<int> input) {
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

void CA::set_rule(vector<int> rule) {
    int i;

    for (i = 0; i < RULELENGTH; ++i)
	_rule[i] = rule[i];
}

/***************************************************************************************/

void CA::apply_rule(real_2d_array& training_data, int data_index) {
    int i, j;
    int rule_index, rule_window[NEIGHBORHOOD];
    int start = -(NEIGHBORHOOD / 2);
    int finish = NEIGHBORHOOD / 2;


    //cout << "Applying rule\n";
    for (; _iter < I; ++_iter) {
	// Could be optimized to only call mod() at ends of row
	for (i = 0; i < WIDTH; ++i) {
	    for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index)
		rule_window[rule_index] = _cell[_iter][mod(i + j, WIDTH)];
	    _cell[_iter + 1][i] = training_data[data_index][i]
		= _rule[base_N_to_dec(rule_window, STATES, NEIGHBORHOOD)];
	}
    }
    // copy last row to initial position
    for (i = 0; i < WIDTH; ++i)
	_cell[0][i] = _cell[_iter][i];
    //cout << "Data_index: " << data_index << endl;
}


/***************************************************************************************/

void CA::build_5_bit_model(real_2d_array& training_data, vector<linearmodel>& output) {
    int time_step, test_set, data_index;
    int i, stop = SEQUENCE_LENGTH * TEST_SETS; 
    int distractor_end = SEQUENCE_LENGTH - 5;
    int model_index;
    ae_int_t info;
    ae_int_t nvars;
    lrreport rep;

    for (model_index = 0; model_index < 3; ++model_index){
	data_index = 0;
	for (test_set = 0; test_set < TEST_SETS; ++test_set) {
	    if (model_index == 2) {
		for (i = 0; i < distractor_end; ++i) 
		    training_data[data_index++][READOUT_LENGTH] = 1;
	    }
	    else if (model_index == 0) {
		for (i = 0; i < distractor_end; ++i) 
		    training_data[data_index++][READOUT_LENGTH] = 0;
	    }
	    // Recall period
	    for (time_step = 0; i < SEQUENCE_LENGTH; ++i, ++time_step) {
                if (model_index == 0) {
		    training_data[data_index++][READOUT_LENGTH] = 
			test_set >> time_step & 1;
		}
		else if (model_index == 1) {
		    training_data[data_index++][READOUT_LENGTH] = 
			!(test_set >> time_step & 1);
		}
		else 
		    training_data[data_index++][READOUT_LENGTH] = 0;
	    }
	}
	cout << "Building linear regression model #" << model_index << endl;
        lrbuild(training_data, SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH, info,
		output[model_index], rep);
	cout << int(info) << endl;

    }
}

/***************************************************************************************/

//training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);
void CA::train_5_bit(real_2d_array& training_data) {
    int time_step ,test_set;
    int output_index, data_index = 0;
    int distractor_end = SEQUENCE_LENGTH - 6;
    vector<int> input(4);

    for (test_set = 0; test_set < TEST_SETS; ++test_set) {
	// Input signal
	for (time_step = 0; time_step < 5; ++time_step) {
	    input[0] = test_set >> time_step & 1;
	    input[1] = !input[0];
	    input[2] = input[3] = 0;
	    set_input(input);
            apply_rule(training_data, data_index++); 
	}
	// Distractor period
	for (; time_step < distractor_end; ++time_step) {
	    input[0] = input[1] = input[3] = 0;
	    input[2] = 1;
	    set_input(input);
            apply_rule(training_data, data_index++); 
	}
	// Distractor signal
	input[0] = input[1] = input[2] = 0;
	input[3] = 1;
	set_input(input);
	apply_rule(training_data, data_index++); 
	++time_step;
        // Recall period
	for (; time_step < SEQUENCE_LENGTH; ++time_step) {
	    input[0] = input[1] = input[3] = 0;
	    input[2] = 1;
	    set_input(input);
            apply_rule(training_data, data_index++); 
	}
    }
    //cout << "Data_index: " << data_index << endl;
}

/***************************************************************************************/

int CA::mod(int x, int y) {
    try {
	if (y < 0)
	    throw NegativeModulusException();
    }
    catch(NegativeModulusException e) {
	cout << "Error: b must be non-negative in a mod b.\n"; 
	exit(1);
    }
    int r = x % y;
    return r < 0 ? r + y : r;
}


/***************************************************************************************/

int CA::base_N_to_dec(int num[], int base, int length) {
    int total = 0;
    int place = 1;
    
    for (int i = length - 1; i >= 0; --i) {
	total += place * num[i];
        place *= base;
    }
    return total;
}






