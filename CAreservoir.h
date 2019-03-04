
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

const std::vector<int> RULE90 = {0,1,0,1,1,0,1,0};

class CA {
    public:
	CA();

    private:
	std::vector<std::vector<int>> _map;  // [R][INPUT_LENGTH]
	std::vector<std::vector<int>> _cell; // [WIDTH][I];
	int _i;
};




#endif
