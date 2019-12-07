
#include <cstdlib>
#include <vector>
#include <iostream>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
//#include <omp.h>
//#include <thread>

using namespace std; 

const int WIDTH = 20;
const int STATES = 3;
const int NEIGHBORHOOD = 3;
const int RUNS = 100;
const int TIME_OUT = 3000;
const int RULELENGTH = pow(STATES, NEIGHBORHOOD);

struct row {
   row* next;
   int* data;
   int length;
};

class CA {
   public:
      CA();
      ~CA();
      void display_row();
      void apply_rule();
      int check_for_repeat(int* trans);
      void load_rule(vector<int> r);
   private: 
      void load_rule(string rule_file);
      vector<int> rule;   
      row* head;
      row* tail;
};


void load_next_rule(ifstream* in, vector<int>& r);
void random_init(int init[]); 
void save_random_rules(); 
int mod(int x, int y); 
int base_N_to_dec(int num[], int base, int length); 
class IncorrectRuleLengthException{};
class NegativeModulusException{};

