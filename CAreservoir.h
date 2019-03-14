
#ifndef CARESERVOIR_H
#define CARESERVOIR_H


int R 		        	= 4;  //8
const int I 			= 2;  //2
const int DIFFUSE_LENGTH 	= 20; //40
const int INPUT_LENGTH 		= 4;  //4
int STATES 		= 3; 
const int NEIGHBORHOOD 		= 3;
int RULELENGTH 		= pow(STATES, NEIGHBORHOOD);
const int WIDTH			= DIFFUSE_LENGTH * R;
const int READOUT_LENGTH	= R * DIFFUSE_LENGTH * I;
const int DISTRACTOR_PERIOD	= 200;
// For 5-bit memory task
const int SEQUENCE_LENGTH	= DISTRACTOR_PERIOD + 10;
const int TEST_SETS		= 32;
const int MAX_THREADS           = 32;

const std::vector<int> RULE102 = {0,1,1,0,0,1,1,0};
const std::vector<int> RULE90 = {0,1,0,1,1,0,1,0};
const std::vector<int> RULE60 = {0,0,1,1,1,1,0,0};
const std::vector<int> RULE153 = {1,0,0,1,1,0,0,1};
const std::vector<int> RULE195 = {1,1,0,0,0,0,1,1};
const std::vector<int> RULE150 = {1,0,0,1,0,1,1,0};
const std::vector<int> RULE1 = {0,0,0,0,0,0,0,1};
const std::vector<int> RULE0 = {0,0,0,0,0,0,0,0};
const std::vector<int> RULE30 = {0,0,0,1,1,1,1,0};
const std::vector<int> RULE180 = {1,0,1,1,0,1,0,0};

const std::vector<int> RULE3_3 = {1,1,1,1,1,1,2,2,2,2,0,0,2,2,2,1,1,2,0,2,1,1,0,0,0,2,2};
//const std::vector<int> RULE3_3 = {2,2,2,2,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2};

class CA {
    public:
	CA();
	void set_input(std::vector<int> input);
	void set_rule(std::vector<int> rule);
	void load_rule(std::string rule_file);
	void apply_rule(alglib::real_2d_array& training_data, int data_index);
        void build_5_bit_model(alglib::real_2d_array& training_data,
		std::vector<alglib::linearmodel>& output);
	void train_5_bit(alglib::real_2d_array& training_data);
	int test_5_bit(alglib::real_2d_array& training_data, 
		std::vector<alglib::linearmodel>& output);
	void check_CA(alglib::real_2d_array& training_data);
	void draw_CA(alglib::real_2d_array& training_data);
	void save_CA(alglib::real_2d_array& training_data);
	int build_SVM_model(alglib::real_2d_array& training_data);
	void set_5_bit_targets();

    private:
        int mod(int x, int y);
	int base_N_to_dec(int num[], int base, int length);
        void call_SVM_functions(int model, int& incorrect, alglib::real_2d_array training_data);

	std::vector<std::vector<int>> 	_map;  // [R][INPUT_LENGTH]
	std::vector<std::vector<int>> 	_cell; // [I + 1][WIDTH];  // [Row][Column]
	std::vector<int> 		_rule;
	int 				_iter;
	std::vector<std::vector<int>> 	_targets;
};

class NegativeModulusException{};
class IncorrectRuleLengthException{};


#endif
