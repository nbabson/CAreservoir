
#ifndef CARESERVOIR_H
#define CARESERVOIR_H


const int R 			= 8;
const int I 			= 2;
const int DIFFUSE_LENGTH 	= 40;
const int INPUT_LENGTH 		= 4;
const int STATES 		= 2; 
const int NEIGHBORHOOD 		= 3;
const int RULELENGTH 		= pow(STATES, NEIGHBORHOOD);
const int WIDTH			= DIFFUSE_LENGTH * R;
const int READOUT_LENGTH	= R * DIFFUSE_LENGTH * I;
const int DISTRACTOR_PERIOD	= 200;
// For 5-bit memory task
const int SEQUENCE_LENGTH	= DISTRACTOR_PERIOD + 10;
const int TEST_SETS		= 32;

const std::vector<int> RULE90 = {0,1,0,1,1,0,1,0};

class CA {
    public:
	CA();
	void set_input(std::vector<int> input);
	void set_rule(std::vector<int> rule);

    private:
	std::vector<std::vector<int>> 	_map;  // [R][INPUT_LENGTH]
	std::vector<std::vector<int>> 	_cell; // [I][WIDTH];  // [Row][Column]
	std::vector<int> 		_rule;
	int 				_iter;
};




#endif
