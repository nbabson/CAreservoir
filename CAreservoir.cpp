
// March 2019
// Celluar Automaton Reservoir

#include <libalglib/stdafx.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <string>
#include <iomanip>
#include <libalglib/dataanalysis.h>
#include "CAreservoir.h"
#include <omp.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <algorithm>


using namespace alglib;
using namespace std;

struct parameters {
    bool draw;
    bool parallel;
    bool build_file;
    bool build_neighborhood_file;
    bool build_density_file;
    bool build_3_state_temp_dens_file;
    bool svm;
    bool uniform;
    bool alglib;
    bool eoc;
    int runs;
    int cores;
    // Only make and draw CA, don't perform benchmark task
    bool only_draw;
    // Perform temporal density and temporal parity benchmarks
    bool temporal_density;
    bool evolve_4_state;
    bool lambda;
    bool lambda_1;
    string rule_file;
};

void parallel_5_bit(int num_tests, int num_threads, string rule_file, bool uniform);
void parallel_SVM(int num_tests, int cores, string rule_file, bool uniform, bool scikit); // Also scikit
void parallel_td(int runs, string rule_file, bool uniform);
void parse_cmd_line(int argc, char** argv, parameters* params);
void usage();
void build_3_state_CA_file(int runs, bool uniform, bool scikit);
void build_5_neighborhood_file(int runs, bool uniform, bool scikit);
void build_3_state_temp_dens_file(int runs, bool uniform);
void dec_to_base_3(vector<int>& result, int num);
bool find_static_CAs(real_2d_array& training_data);
void random_rule(vector<int>& rule);
void test_2_state_density_rules(bool uniform, bool scikit);
void only_draw(bool uniform, string rule_file, int height);
float lambda(vector<int> rule); 
void calculate_lambdas();
void stochastic_3_state_temp_dens(int runs);
void temporal_density(bool uniform, bool draw, string rule_file, int* scores,
        string rule1, string rule2); 
void random_rule_string(string& rule);
void build_4_state_file(int runs);
void temp_dens_make_dens_rules(); 
void dec_to_base_N(int num, int base, vector<int>& ans); 
void eoc();
void save_RA_format(string rulefile);
void count_lambda();
void make_lambda_1_rules();

int main(int argc, char **argv) {
    parameters params;

    parse_cmd_line(argc, argv, &params);    
    srand(time(NULL));
    
    //count_lambda();
    /*
    vector<int> r;
    r.resize(RULELENGTH);
    random_rule(r);
    FILE* f_out = fopen("rand_3_state.txt", "a");
    for (int i = 0; i < RULELENGTH; ++i)
        fprintf(f_out, "%d", r[i]);
    fclose(f_out);
    exit(0);
    */
    //save_RA_format("t1.txt");
    //save_RA_format("rand_3_state.txt");
    //exit(0);

    try {
        if (params.lambda_1)
           make_lambda_1_rules();

        else if (params.only_draw) {
            if (params.temporal_density)
                only_draw(params.uniform, params.rule_file, TEMP_DENS_TRAINING_SIZE);
            else
                only_draw(params.uniform, params.rule_file, SEQUENCE_LENGTH * TEST_SETS);
        }
        else if (params.lambda)
           calculate_lambdas();
        else if (params.eoc)
           eoc();
        else if (params.build_file) {
	    if (params.alglib)
	        build_3_state_CA_file(params.runs, params.uniform, false);
	    else
	        build_3_state_CA_file(params.runs, params.uniform, true);
	}
        else if (params.evolve_4_state) {
            build_4_state_file(params.runs);
        }
        else if (params.build_neighborhood_file) {
	    if (params.alglib)
		build_5_neighborhood_file(params.runs, params.uniform, false);
	    else {
                if (params.temporal_density)
                    stochastic_3_state_temp_dens(params.runs);
                else    
		    build_5_neighborhood_file(params.runs, params.uniform, true);
            }
	}
        else if (params.build_3_state_temp_dens_file) {
            //build_3_state_temp_dens_file(params.runs, params.uniform);
            stochastic_3_state_temp_dens(params.runs);
        }
	else if (params.build_density_file) {
	    if (params.alglib)
                test_2_state_density_rules(params.uniform, false);
	    else
                if (params.temporal_density)
                    temp_dens_make_dens_rules(); 
                else     
                    test_2_state_density_rules(params.uniform, true);
	}
	else if (params.eoc) {
            if (params.alglib) {}
	    else {}
	}
	else if (params.parallel) {
	    if (params.svm)
	        parallel_SVM(params.runs, params.cores, params.rule_file, params.uniform, false);
	    else if (params.alglib)
		parallel_5_bit(params.runs, params.cores, params.rule_file, params.uniform);
	    else {  // scikit
                if (params.temporal_density)
                    parallel_td(params.runs, params.rule_file, params.uniform);
                else
	            parallel_SVM(params.runs, params.cores, params.rule_file, params.uniform, true);
            }
	}
        else if (params.temporal_density) {
            int scores[2];
            temporal_density(params.uniform, params.draw, params.rule_file, scores, "-1", "-1"); 
        }
	else {
            CA ca(params.uniform, false);
	    if (params.uniform)
		ca.load_rule(params.rule_file);
	    else
	        ca.load_two_rules(params.rule_file);
	    real_2d_array training_data;
	    vector<linearmodel> output(3);
	    // Add one for target
	    training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);
	    cout << "Building training data\n";
	    ca.train_5_bit(training_data, params.uniform);
	    ca.set_5_bit_targets();
	    if (params.svm)
		ca.build_SVM_model(training_data);
	    else if (params.alglib) {
		ca.build_5_bit_model(training_data, output);
		ca.test_5_bit(training_data, output);
	    }
	    else {  // scikit
		ca.build_scikit_model(training_data);
	    }
	    if (params.draw)
		ca.save_CA(training_data, SEQUENCE_LENGTH * TEST_SETS);
	}
        int sys = system("make clean");
        puts("make clean");
    }
    catch(IncorrectRuleLengthException e)
    {
        cout << "Error: rule length does not match number of states and neighborhood.\n";
    }	    
    catch (NonUniformRuleFileFormatException e)
    {
	cout << "Error: first non-uniform CA rule must be followed by ':'.\n";
    }
    return 0;
}

/***************************************************************************************/

void parse_cmd_line(int argc, char** argv, parameters* params) {
    int arg_index = 1;
    int max_arg = argc - 2;
    bool error;

    params -> lambda = false;
    params -> lambda_1 = false;
    params -> draw = false;
    params -> parallel = false;
    params -> build_file = false;
    params -> svm = false;
    params -> alglib = false;
    params -> uniform = true;
    params -> build_neighborhood_file = false; 
    params -> build_density_file = false;
    params -> eoc = false;
    params -> only_draw = false;
    params -> temporal_density = false;
    params -> build_3_state_temp_dens_file = false;;
    params -> evolve_4_state = false;

    if (argc < 3) 
	usage();
    while (arg_index < max_arg) {
	error = true;
	if (!strcmp(argv[arg_index], "-d")) {
	    params -> draw = true;
	    error = false;
	}
        else if (!strcmp(argv[arg_index], "-ld")) {
            LONG_DRAW = true;
            params -> draw = true;
            error = false;
        }
        else if (!strcmp(argv[arg_index], "-p")) {
	    params -> parallel = true;
            params -> runs = atoi(argv[++arg_index]);
	    params -> cores = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-r")) {
	    R = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-i")) {
	    I = atoi(argv[++arg_index]);
	    error = false;
	}
        else if (!strcmp(argv[arg_index], "-ev")) {
            params -> evolve_4_state = true;
            params -> runs = atoi(argv[++arg_index]);
            error = false;
        }
	else if (!strcmp(argv[arg_index], "-lr")) {
	    params -> alglib = true;
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-svm")) {
	    params -> svm = true;
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-dl")) {
            DIFFUSE_LENGTH = atoi(argv[++arg_index]);
            PARITY_INPUT_LENGTH = DIFFUSE_LENGTH / 4;
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-pl")) {
            PARITY_INPUT_LENGTH = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-ts")) {
            TEMP_DENS_TRAINING_SIZE = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-ne")) {
	    NEIGHBORHOOD = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-bf")) {
	    params -> build_file = true;
	    params -> runs = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-bn")) {
	    params -> build_neighborhood_file = true;
	    params -> runs = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-btd")) {
	    params -> build_3_state_temp_dens_file = true;
	    params -> runs = atoi(argv[++arg_index]);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-nd")) {
            NU_RULES_SAME_TYPE = false;
	    NEIGHBORHOOD2 = atoi(argv[++arg_index]);
	    STATES2 = atoi(argv[++arg_index]);
	    RULELENGTH2 = pow(STATES2, NEIGHBORHOOD2);
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-2pd")) {
            RULE_2_POP_DENS = true;
	    STATES2 = 2;
	    NEIGHBORHOOD2 = 5;
	    NU_RULES_SAME_TYPE = false;
	    RULELENGTH2 = STATES2 * NEIGHBORHOOD2;
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-bd")) {
	    params -> build_density_file = true;
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-pd")) {
	    DENSITY_RULE = true;
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-nu")) {
	    params -> uniform = false;
	    error = false;
	}
	else if (!strcmp(argv[arg_index], "-eoc")) {
	    params -> eoc = true;
	    error = false;
	}
        else if (!strcmp(argv[arg_index], "-od")) {
            params -> only_draw = true;
            error = false;
        }
        else if (!strcmp(argv[arg_index], "-td")) {
            params -> temporal_density = true;
            TD = true;
            error = false;
        }
        else if (!strcmp(argv[arg_index], "-n")) {
            N = atoi(argv[++arg_index]);
            error = false;
        }
        else if (!strcmp(argv[arg_index], "-ai")) {
            OV = false;
            error = false;
        }
        else if (!strcmp(argv[arg_index], "-de")) {
            DELAY = atoi(argv[++arg_index]);
            error = false;
        }
        else if (!strcmp(argv[arg_index], "-time")) {
           TIME = true;
           error = false;
        }
        else if (!strcmp(argv[arg_index], "-lambda")) {
           params -> lambda = true;
           error = false;
        }
        else if (!strcmp(argv[arg_index], "-l1")) {
           params -> lambda_1 = true;
           error = false;
        }
        else if (!strcmp(argv[arg_index], "-eoc")) {
           params -> eoc = true;
           error = false;
        }
	++arg_index;
	if (error)
	    usage();
    }
    if (arg_index + 2 != argc)
	usage();
    WIDTH = DIFFUSE_LENGTH * R;
    READOUT_LENGTH = R * DIFFUSE_LENGTH * I;
    if (!isdigit(argv[arg_index][0]))
	usage();
    STATES = atoi(argv[arg_index]);
    if (DENSITY_RULE)
	RULELENGTH = STATES * NEIGHBORHOOD;
    else
	RULELENGTH = pow(STATES, NEIGHBORHOOD);
    if (NU_RULES_SAME_TYPE)
        RULELENGTH2 = RULELENGTH;
    ++arg_index;
    params -> rule_file = argv[arg_index];
}

/***************************************************************************************/

void usage() {
    cout << "Usage: CAreservoir [options] <# of states> <rule file>\n";
    cout << "Options:\n";
    cout << "-svm             -> use Support Vector Machine (instead of SciKit)\n";
    cout << "-lr              -> use AlgLib linear regression (instead of SciKit)\n";
    cout << "-d               -> save CA to ca.txt and draw in ca.ppm\n";
    cout << "-ld              -> change draw length to long\n";
    cout << "-p <int1> <int2> -> parallel: <int1> runs on up to <int2> cores\n"; 
    cout << "-r <int>         -> change R, # of reservoirs\n"; 
    cout << "-i <int>         -> change I, # of CA iterations\n"; 
    cout << "-dl <int>        -> change DIFFUSION_LENGTH, size of reservoirs\n";
    cout << "-ai              -> change mode input from overwrite to additive\n";
    cout << "-pd              -> population density rule (2 state for now)\n"; 
    cout << "-bf <int>        -> build 3 state CA rule file, # of runs\n";
    cout << "-bn <int>        -> build neighborhood 5 CA rule file, # of runs\n";
    cout << "-bd              -> build density rule file (2 state, 5 neighbor) \n";
    cout << "-btd <int>       -> build temporal density three state rule file, # of runs\n";
    cout << "-eoc             -> build random Edge of Chaos rule\n";
    cout << "-ne <int>        -> change NEIGHBORHOOD size from 3 to another odd number\n";
    cout << "-od              -> only create and draw CA -- don't perform 5-bit task\n";
    cout << "-ev <int>        -> build file of evolved 4 state rules, # of runs\n";
    cout << "-td              -> temporal density benchmark task\n";  
    cout << "-n <int>         -> change N, size of window for temporal density benchmark\n";
    cout << "-de <int>        -> change DELAY for temporal density benchmark\n";
    cout << "-pl <int>        -> change input length for temporal density and parity benchmarks\n";
    cout << "-ts <int>        -> change training/test set size for temporal benchmarks\n";
    cout << "-nu              -> non-uniform reservoir rules, rule file should have\n";
    cout << "                    2 rules on successive lines where first line ends with ':'\n";
    cout << "-nd <int1> <int2>-> non-uniform rules of different types, <int1> NEIGHBORHOOD for second rule\n";
    cout << "                    <int2> STATES for second rule\n";
    cout << "-2pd             -> second rule only of non-uniform reservoir is a population density rule\n";
    exit(0);
}


/***************************************************************************************/

void make_lambda_1_rules() {
   ofstream out;
   int correct;
   out.open("lambda_1_four_state.txt", ofstream::out | ofstream::app);
   //out.open("lambda_1_five_state.txt", ofstream::out | ofstream::app);
   #pragma omp parallel
   {
      omp_set_num_threads(40);
      #pragma omp for nowait 
      for (int j = 0; j < 400; ++j) {
         CA ca(true, false);
         for (int i = 0; i < RULELENGTH; ++i) {
            //ca._rule[i] = (rand() % 3) + 2; // five state
            ca._rule[i] = (rand() % 2) + 2; // four state
            //out << (rand() % 2) + 1; // three state
         }

         real_2d_array training_data;
         training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);

         ca.train_5_bit(training_data, true);
         ca.set_5_bit_targets();
         correct = ca.build_scikit_model(training_data);
         if (correct == 0) {
            #pragma omp critical
            {
               for (int k = 0; k < RULELENGTH; ++k) 
                  out << ca._rule[k];
               out << "   incorrect: " << correct << endl;  
            }
         }
      }
   }
   out.close();
}

/***************************************************************************************/
void save_RA_format(string rulefile) {
   CA ca(true, false);
   int row;
   int a = 0, b = 0, c = 0;

   ca.load_rule(rulefile); 

   ofstream out;
   //string outfile = "good_3_3_rule.txt";
   string outfile = "bad2.txt";

   out.open(outfile, ofstream::out);
   for (row = 0; row < RULELENGTH; ++row) {
      out << " " << a << " " << b << " " << c << " " << 
         ca._rule[RULELENGTH - row - 1] << endl;
      ++c;
      if (c == STATES) {
         c = 0;
         ++b;
         if (b == STATES) {
           b = 0;
           ++a;
         }
      } 
   }
   out.close();
}

/***************************************************************************************/

void count_lambda() {
   ifstream in;
   string line;
   string in_file = "lambda_3_state_rules.txt";
   //string in_file = "lambda_rules.txt";
   int i, count = 0;
   float lambda[11] = {0}, l, total = 0;
   size_t position, sz;

   in.open(in_file.c_str(), ifstream::in);
   for (i = 0; i < 73; ++i)
      getline(in, line);

   getline(in, line);
   while (!in.eof()) {
      if (line[0] != '<') {
         position = line.find_last_of(" ");
         l = stof(line.substr(position + 1), &sz);
         //cout << l << "  ";
         total += l;
         ++count;
         ++lambda[(int)(10*l)];
      }
      if (count > 100) 
         break;
      getline(in, line);
   }
   for (i = 1; i < 11; ++i)
      cout << "lambda " << ((float)i)/10 << ":   " << lambda[i] << endl;
   cout << "Total: " << count << endl;
   cout << "Average: " << total/(float)count << endl;


}

/***************************************************************************************/

void eoc() {
   //cout << "In EOC\n";
   ofstream out;
   int num_tests = 1; //100;
   int success[11] = {0};

   out.open("lambda_3_state_rules.txt", ofstream::out | ofstream::app);

   #pragma omp parallel
   {
      omp_set_num_threads(40);
      #pragma omp for collapse(2) nowait 
      for (int lamb = 1; lamb < 11; ++lamb)
      {
         for (int j  = 0; j < num_tests; ++j) {
            float lambda = ((float)lamb) / 10;
            //cout << "--->>> " << lambda << " <<<---\n";
            int quiescent = RULELENGTH - (int)round(lambda * RULELENGTH);
            int k, correct; 
            CA ca(true, false);
            real_2d_array training_data;
            training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);

            for (k = 0; k < quiescent; ++k)
               ca._rule[k] = 0;
            for (; k < RULELENGTH; ++k)
               ca._rule[k] = 1 + (rand() % (STATES - 1));
            random_shuffle(ca._rule.begin(), ca._rule.end());
            //for (k = 0; k < RULELENGTH; ++k)
            //   cout << ca._rule[k];
            //cout << endl;
            ca.train_5_bit(training_data, true);
            ca.set_5_bit_targets();
            correct = ca.build_scikit_model(training_data);
            if (correct < 6) {
               #pragma omp critical
               {
                  ++success[lamb];
                  for (int k = 0; k < RULELENGTH; ++k) 
                     out << ca._rule[k];
                  out << "   incorrect: " << correct << "    lambda: " <<
                              lambda << endl;  
               }
            }
         }
      }
   }
   out.close();
   cout << "Success: \n";
   for (int i = 1; i < 11; ++i) 
      cout << "Lambda " << (float)i/10 << ":      " << success[i] << endl;
}

/***************************************************************************************/

void calculate_lambdas() {
   ifstream in;
   ofstream out;
   int i, j;
   vector<int> r(RULELENGTH);
   string rule_file = "rule.txt";
   string dest_file = "three_state_lambda.txt";
   //string rule_file = "four_state_rules_smallest_largest.txt";
   //string dest_file = "four_state_lambda.txt";
   //string rule_file = "five_state_rules.txt";
   //string dest_file = "five_state_lambda.txt";
   //string rule_file = "neighborhood_5_rules.txt";
   //string dest_file = "neighborhood_five_lambda.txt";
   char x;
   float l;
   float count = 0;
   float ave = 0;

   in.open(rule_file.c_str(), ifstream::in);
   //out.open(dest_file.c_str(), ofstream::out);
   
   in >> x;
   while (!in.eof()) {
      r[0] = x - 48;
      for (i = 1; i < RULELENGTH; ++i) {
         in >> x;
         r[i] =  x - 48;
      }
      in.ignore(numeric_limits<streamsize>::max(), '\n');
      /*
      while (x != '\n' && !in.eof()) {
         in >> x;
         cout << x;
      }
      */
      for (i = 0; i < RULELENGTH; ++i) {
          cout << r[i];
          //out << r[i];
      }
      l = lambda(r);
      cout << '\t'<< l << endl;
      //out << '\t' << l << endl;
      in >> x;
      ++count;
      ave += l;
      //if (count == 32)         // four state
      //if (count == 26)         // five state
      //if (count == 37)         // neighborhood five
      //   break;
   }
   cout << "Average: " << ave / count;
   //out << "Average: " << ave / count;
   cout << endl << endl << "Random rules: " <<  endl;
   count = 0; 
   ave = 0;
   for (i = 0; i < 30; ++i) {
      random_rule(r);
      for (j = 0; j < RULELENGTH; ++j)
         cout << r[j];
      l = lambda(r);
      cout << '\t' << l << endl;
      ++count;
      ave += l;
   }
   cout << "Average: " << ave / count << endl;

   in.close();
   //out.close();
}


/***************************************************************************************/

float lambda(vector<int> rule) {
   int s[STATES] = {0};
   int i, max = 0;

   for (i = 0; i < RULELENGTH; ++i)
      s[rule[i]]++;
   for (i = 1; i < STATES; ++i)
      if (s[i] > s[max])
      //if (s[i] < s[max])
        max = i;
   //cout << "    " << s[max];
   //return (RULELENGTH - s[max]) / (float)RULELENGTH; 
   cout << "    " << s[0];   // Use state 0 as quiescent
   return (RULELENGTH - s[0]) / (float)RULELENGTH; 
   
}

/***************************************************************************************/

void only_draw(bool uniform, string rule_file, int height) {

    CA ca(uniform, false);
    if (uniform)
        ca.load_rule(rule_file);
    else
        ca.load_two_rules(rule_file);
    real_2d_array training_data;
    // Add one for target
    training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);
    // Random initial generation
    vector<int> input = {0,1,0,1};
    ca.set_input(input);
    // Middle cell only
    //ca.set_middle();

    ca.check_CA(training_data, uniform);

    ca.save_CA(training_data, height);
}

/***************************************************************************************/

void CA::set_middle() {
    _iter = 0;
    _cell[0][200] = 1; //READOUT_LENGTH / 2] = 1;
}

/***************************************************************************************/

void temp_dens_make_dens_rules() {
    int start = 0;
    int stop =  1024; // 1024; 
    int success = 0;
    int td = 0, tp = 0;
    ofstream out;
    int good_CA_count = 0;

    out.open("temp_dens_density_rules.txt", ofstream::out | ofstream::app);
    try {
	if (STATES != 2 || NEIGHBORHOOD != 5) throw Expect2State5NeighborhoodException();
    }
    catch(Expect2State5NeighborhoodException e)
    {
	cout << "Error: STATES must be 2 and NEIGHBORHOOD 5 for -bd option.\n"; 
	exit(1);
    }
    int i;  
#pragma omp parallel
    {
        #pragma omp for nowait
	for (i = start; i < stop; ++i) {
            int scores[2];
            string rulestring;
            vector<int> rule(RULELENGTH, 0);
            dec_to_base_N(i, 2, rule);
            for (int j = 0; j < RULELENGTH; ++j) 
                rulestring += rule[j] + 48;
            temporal_density(true, false, "-1", scores, rulestring, "");
            cout << "Scores: "  << scores[0] << "\t" << scores[1] << endl;
            if (scores[0] == 0 || scores[1] == 0) {
                #pragma omp critical
                {
                    ++good_CA_count;
                    out << rulestring << " " << scores[0] << "/" << scores[1] << endl;
                }
            }
        }
        
    }
    cout << "Good CAs: " << good_CA_count << endl;    
    out.close();
}

/***************************************************************************************/

void test_2_state_density_rules(bool uniform, bool scikit) {
    ofstream out;
    int start = 0;
    int stop =  1024;//1024; 
    int good_CA_count = 0;
    int errors, reject_CAs = 0;

    out.open("density_rules2.txt", ofstream::out | ofstream::app);
    try {
	if (STATES != 2 || NEIGHBORHOOD != 5) throw Expect2State5NeighborhoodException();
    }
    catch(Expect2State5NeighborhoodException e)
    {
	cout << "Error: STATES must be 2 and NEIGHBORHOOD 5 for -bd option.\n"; 
	exit(1);
    }
     vector<int> input = {0,1,0,1};
    int i;  
#pragma omp parallel
    {
        #pragma omp for nowait
	for (i = start; i < stop; ++i) {
            CA ca;
            int j; //, epoch, data_index = 0;

            real_2d_array training_data;
            vector<linearmodel> output(3); 

            training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH);
            vector<int> rule(RULELENGTH, 0);
            ca.dec_to_base_N(i, 2, rule);
            ca.set_rule(rule);
            ca.set_input(input);
            
            //   for (int k = 0; k < RULELENGTH; ++k)
            //     cout << rule[k];
            //   cout << endl;
            
            cout << "Making rule " << i << ".\n";
                ca.check_CA(training_data, uniform);
                if (!find_static_CAs(training_data)) {
            //ca.save_CA(training_data,WIDTH);
            ca.train_5_bit(training_data, uniform);
            if (scikit) {
                ca.set_5_bit_targets();
                errors = ca.build_scikit_model(training_data);
            }
            else {
                ca.build_5_bit_model(training_data, output);
                errors = ca.test_5_bit(training_data, output);
            }
            if (errors < 1) {
#pragma omp critical
                {
                    ++good_CA_count;
                    for (j = 0; j < RULELENGTH; ++j)
                        out << rule[j];
                    out << " " << errors << "\n";
                }
            }
                }
                 else {
                 #pragma omp critical
                            {
                                ++reject_CAs;
                            }
                        }
        }
    }
    cout << "Good CAs: " << good_CA_count << endl;
    cout << "Rejected as static: "  << reject_CAs << "\n";
    out.close();

}

/***************************************************************************************/

void parallel_td(int runs, string rule_file, bool uniform) {
   int success = 0;
   int td = 0, tp = 0;

    srand(time(NULL));
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < runs; ++i) 
	{
            int scores[2];
            cout << "Building training data\n";
            temporal_density(uniform, false, rule_file, scores, "-1", "-1");
            if (scores[0] + scores[1] == 0) {
                #pragma omp critical
                {
                    ++success;
                    ++td;
                    ++tp;
                }
            }
            else if (scores[0] == 0) {
                #pragma omp critical
                {
                    ++td;
                }
            }
            else if (scores[1] == 0) {
                #pragma omp critical
                {
                    ++tp;
                }
            }
        }
    }
    cout << "Successful tests: " << success << ", out of " << runs << "." << endl;
    cout << "Successful temporal density: " << td << ", out of " << runs << "." << endl;
    cout << "Successful temporal parity: " << tp << ", out of " << runs << "." << endl;
}


/***************************************************************************************/

// Build and test 5 bit task support vector machine
// or SciKit linear regression CAs in parallel
void parallel_SVM(int num_tests, int cores, string rule_file, bool uniform, bool scikit) {
    int success = 0;
    //omp_set_nested(1);
    srand(time(NULL));
    // Don't exceed number of cores
    omp_set_num_threads(cores);
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < num_tests; ++i) 
	{
	    CA ca(uniform, false);
	    real_2d_array training_data;
	    try {
		if (uniform)
		    ca.load_rule(rule_file);
		else
	            ca.load_two_rules(rule_file);
		training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);

		cout << "Building training data\n";
		ca.train_5_bit(training_data, uniform);
		ca.set_5_bit_targets();
		if (scikit) {
		    if (ca.build_scikit_model(training_data) == 0) {
			#pragma omp critical
			{
			    ++success;
			}
		    }
		}
		else {
		    if (ca.build_SVM_model(training_data) == 0) {
			#pragma omp critical
			{
			    ++success;
			}
		    }
		}
	    }
	    catch(IncorrectRuleLengthException e)
	    {
		cout << "Error: rule length does not match number of states and neighborhood.\n";
		exit(1);
	    }	    
	    catch (NonUniformRuleFileFormatException e)
	    {
		cout << "Error: first non-uniform CA rule must be followed by ':'.\n";
		exit(1);
	    }
	}
    }
    cout << "Successful tests: " << success << ", out of " << num_tests << "." << endl;
}

/***************************************************************************************/

// Parallel linear regression tests
void parallel_5_bit(int num_tests, int num_threads, string rule_file, bool uniform) { 
    int success = 0;
    omp_set_nested(1);
    // Don't exceed number of cores
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < num_tests; ++i) 
	{
	    try {
		CA ca(uniform, false);
		real_2d_array training_data;
		vector<linearmodel> output(3);
		if (uniform)
		    ca.load_rule(rule_file);
		else
		    ca.load_two_rules(rule_file);
		training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);
		cout << "Building training data\n";
		ca.train_5_bit(training_data, uniform);
		cout << "Building regression models\n";
		ca.build_5_bit_model(training_data, output);
		if (ca.test_5_bit(training_data, output) == 0) {
		    #pragma omp critical
		    {
			++success;
		    }
		}
	    }
	    catch(IncorrectRuleLengthException e)
	    {
		cout << "Error: rule length does not match number of states and neighborhood.\n";
		exit(1);
	    }	    
	    catch (NonUniformRuleFileFormatException e)
	    {
		cout << "Error: first non-uniform CA rule must be followed by ':'.\n";
		exit(1);
	    }
	}
    }
    cout << "Successful tests: " << success << ", out of " << num_tests << "." << endl;
}

/***************************************************************************************/

// CA constructor sets random input mapping for each of R subreservoirs
CA::CA() {
    bool unique;
    int i, j, k;

    _iter = 0;
    _map.resize(R, vector<int>(INPUT_LENGTH));
    _cell.resize(I + 1, vector<int>(WIDTH));
    _rule.resize(RULELENGTH);
    _targets.resize(3, vector<int>(SEQUENCE_LENGTH * TEST_SETS));
    // Initialize first row with 0s for 2 STATE CA
    if (STATES == 2) {
	for (i = 0; i < WIDTH; ++i)
	    _cell[0][i] = 0;
    }
    // Initialize with largest state #
    else {
	for (i = 0; i < WIDTH; ++i)
	    _cell[0][i] = STATES - 1;
    }
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

// CA constructor sets random input mapping for each of R subreservoirs
// temp_dens: 0 -> 5-bit task
//            1 -> temporal density/parity
CA::CA(bool uniform, int temp_dens) {
    bool unique;
    int i, j, k, input_length;

    _iter = 0;
    input_length = temp_dens != 0? PARITY_INPUT_LENGTH: INPUT_LENGTH;
    _map.resize(R, vector<int>(input_length));
    _cell.resize(I + 1, vector<int>(WIDTH));
    _rule.resize(RULELENGTH);
    if (!uniform)
        _rule2.resize(RULELENGTH2);
    if (temp_dens)
        _targets.resize(2, vector<int>(TEMP_DENS_TRAINING_SIZE));
    else   
        _targets.resize(3, vector<int>(SEQUENCE_LENGTH * TEST_SETS));
    // Initialize first row with 0s for 2 STATE CA
    if (STATES == 2) {
	for (i = 0; i < WIDTH; ++i)
	    _cell[0][i] = 0;
    }
    // Initialize with largest state #
    else {
	for (i = 0; i < WIDTH; ++i)
	    _cell[0][i] = STATES - 1;
    }
    if (!NU_RULES_SAME_TYPE) {
       if (STATES2 != STATES) {
	  if (STATES2 == 2) {
	     for (i = WIDTH/2; i < WIDTH; ++i)
		_cell[0][i] = 0;
	  }
	  else {
	     for (i = WIDTH/2; i < WIDTH; ++i) {
		_cell[0][i] = STATES2 - 1;
             }
	  }
       }
    }

    for (i = 0; i < R; ++i) {
	for (j = 0; j < input_length; ++j) {
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

void CA::set_rule(string rule) {
    for (int i = 0; i < RULELENGTH; ++i)
        _rule[i] = rule[i] - 48;
}

/***************************************************************************************/

void CA::set_rule2(string rule) {
    for (int i = 0; i < RULELENGTH2; ++i)
        _rule2[i] = rule[1] - 48;
}

/***************************************************************************************/

void CA::print_first_row() {
    cout << endl;
    for (int i = 0; i < 480; ++i)
        cout << _cell[0][i];
    cout << endl;
}

/***************************************************************************************/

// Changed to use single input size and a single CA for both tasks

// Perform temporal density and temporal parity tasks
void temporal_density(bool uniform, bool draw, string rule_file, int* scores,
        string rule1, string rule2) {
    CA ca(uniform, 1); 
    int i, count = 0, ones = 0;
    int incorrect = 0;

    if (rule1 == "-1") {
        if (uniform) 
            ca.load_rule(rule_file);
        else 
            ca.load_two_rules(rule_file);
    }
    else {
        ca.set_rule(rule1);
        if (!uniform)
            ca.set_rule2(rule2);
        
    }
    real_2d_array training_data;
    training_data.setlength(TEMP_DENS_TRAINING_SIZE, READOUT_LENGTH);
    ca.train_temp_density(training_data, uniform);
    scores[0] = scores[1] = 0;
    ca.temp_density_regression(scores, training_data, uniform); 
    cout << "\nTemporal Density Incorrect: " << scores[0] << endl;
    cout << "\nTemporal Parity Incorrect: " << scores[1] << endl;
    if (draw) {
        ca.save_CA(training_data, TEMP_DENS_TRAINING_SIZE);
    }
}


/***************************************************************************************/

// Build scikit linear regression models for temporal density and temporal parity tasks
int CA::temp_density_regression(int* scores, real_2d_array& training_data,
        bool uniform) { 
    string build_model = "python3 build_density_model.py sk";
    string data_file = "sk";
    string output_file = "sk_results";
    string output_file2 = "sk_results2";
    string tags_file = "sk_density_tags";
    string tags_file2 = "sk_density_tags2";
    int tid = omp_get_thread_num();
    int sk_tag, system_result, i;
    ofstream out;
    ifstream in;
    float result;

    // Write temporal density and temporal parity train data
    out.open((data_file + to_string(tid) + ".csv").c_str(), ofstream::out);
    // Ignore first WINDOW + DELAY generations of CA
    for (i = N + DELAY - 1; i < TEMP_DENS_TRAINING_SIZE - DELAY; ++i) {
       for (int j = 0; j < READOUT_LENGTH - 1; ++j) {
           out << training_data[i][j] << " ";
           }
       // Last data w/o space
       out << training_data[i][READOUT_LENGTH - 1] << endl;
    }
    out.close(); 
    
    // Write temporal density tags
    out.open((tags_file + to_string(tid) + to_string(0) + ".txt").c_str(), ofstream::out);
    for (i = N + DELAY - 1; i < TEMP_DENS_TRAINING_SIZE - DELAY - 1; ++i) {
        out << _targets[0][i] << " ";
    }
    out << _targets[0][TEMP_DENS_TRAINING_SIZE - DELAY - 1];
    out.close();

    // Write temporal parity tags
    out.open((tags_file2 + to_string(tid) + to_string(1) + ".txt").c_str(), ofstream::out);
    for (i = N + DELAY - 1; i < TEMP_DENS_TRAINING_SIZE - DELAY - 1; ++i) {
        out << _targets[1][i] << " ";
    }
    out << _targets[0][TEMP_DENS_TRAINING_SIZE - DELAY - 1];
    out.close();

    // Rerun CA for test data, write to sk_density_test[tid].csv, put new targets in _target
    // Reset first generation of CA
    for (i = 0; i < WIDTH; ++i) {
        if (STATES == 2)
            _cell[0][i] = 0;
        else 
            _cell[0][i] = STATES - 1;
    }
    if (!NU_RULES_SAME_TYPE) {
       if (STATES2 != STATES) {
	  if (STATES2 == 2) {
	     for (i = WIDTH/2; i < WIDTH; ++i)
		_cell[0][i] = 0;
	  }
	  else {
	     for (i = WIDTH/2; i < WIDTH; ++i) {
		_cell[0][i] = STATES2 - 1;
             }
	  }
       }
    }
    train_temp_density(training_data, uniform);

    // Write temporal density and temporal parity test data
    out.open(("sk_density_test" + to_string(tid) + ".csv").c_str(), ofstream::out);
    // Ignore first WINDOW + DELAY generations of CA
    for (i = N + DELAY - 1; i < TEMP_DENS_TRAINING_SIZE - DELAY; ++i) {
       for (int j = 0; j < READOUT_LENGTH - 1; ++j) {
           out << training_data[i][j] << " ";
       }
       // Last data w/o space
       out << training_data[i][READOUT_LENGTH - 1] << endl;
    }
    out.close(); 

    // Build and test model for temporal density task
    system_result = system((build_model + to_string(tid) + ".csv sk_density_tags" + 
                to_string(tid) + to_string(0) + ".txt sk_density_test" + to_string(tid) + ".csv sk_results"
                + to_string(tid) + ".csv").c_str());
    puts((build_model + to_string(tid) + ".csv sk_density_tags" + to_string(tid) + to_string(0) +
                ".txt sk_density_test" + to_string(tid) + ".csv  sk_results" +
                to_string(tid) + ".csv").c_str());
    in.open((output_file + to_string(tid) + ".csv").c_str(), ifstream::in);
    for (i = N + DELAY - 1; i < TEMP_DENS_TRAINING_SIZE - DELAY; ++i) {
        in >> result;
        if ((result < .5 && _targets[0][i] == 1) || (result >= .5 && _targets[0][i] == 0)) {
            #pragma omp critical
            {
                ++scores[0];
            }
        }
    }
    in.close();

    // Build and test model for temporal parity task
    system_result = system((build_model + to_string(tid) + ".csv sk_density_tags2" + 
                to_string(tid) + to_string(1) + ".txt sk_density_test" + to_string(tid) + ".csv sk_results2"
                + to_string(tid) + ".csv").c_str());
    puts((build_model + to_string(tid) + ".csv sk_density_tags2" + to_string(tid) + to_string(1) +
                ".txt sk_density_test" + to_string(tid) + ".csv  sk_results2" +
                to_string(tid) + ".csv").c_str());
    in.open((output_file2 + to_string(tid) + ".csv").c_str(), ifstream::in);
    for (i = N + DELAY - 1; i < TEMP_DENS_TRAINING_SIZE - DELAY; ++i) {
        in >> result;
        if ((result < .5 && _targets[1][i] == 1) || (result >= .5 && _targets[1][i] == 0)) {
            #pragma omp critical
            {
                ++scores[1];
            }
        }
    }
    in.close();
}

/***************************************************************************************/

void CA::train_temp_density (real_2d_array& training_data, bool uniform) {
    int i, ones = 0;
    int data_index = 0;
    int input_length;
    vector<int> inputs(TEMP_DENS_TRAINING_SIZE);

    input_length =  PARITY_INPUT_LENGTH;
    vector<int> input(input_length);
    for (i = 0; i < N; ++i) {
        inputs[i] = rand() % 2;
        for (int j = 0; j < input_length; ++j)
            input[j] = inputs[i];
        if (inputs[i] == 1)
            ++ones;
        set_parity_input(input);
        if (uniform)
            apply_rule(training_data, data_index++);
        else
            apply_two_rules(training_data, data_index++);
    }
    _targets[0][N + (DELAY - 1)] = 2 * ones > N;
    // Removed delay from parity task
    _targets[1][N  - 1] = ones % 2 == 1;
    for (; i < TEMP_DENS_TRAINING_SIZE - DELAY; ++i) {
        inputs[i] = rand() % 2;
        for (int j = 0; j < input_length; ++j)
            input[j] = inputs[i];
        if (inputs[i] == 1)
            ++ones;
        if (inputs[i - N] == 1)
            --ones;
        // Temporal density target
        _targets[0][i + DELAY] = 2 * ones > N;
        _targets[1][i] = ones % 2 == 1;
        set_parity_input(input);
        if (uniform)
            apply_rule(training_data, data_index++);
        else
            apply_two_rules(training_data, data_index++);
    }
    
    /*
    // Test Output
    for (i = 0; i < TEMP_DENS_TRAINING_SIZE - DELAY; ++i) {
        cout << i << "\t" << inputs[i] << "\t" << _targets[0][i] << 
            "\t" << _targets[1][i] << endl; //"\t" << "ones: " << one_count[i] <<
         //  "\t" << one_count[i - DELAY] <<  endl;
    }
    */
}

/***************************************************************************************/

// Load CA rule from rule_file
void CA::load_rule(string rule_file) {
    ifstream in;
    char x;

    in.exceptions(ifstream::failbit | ifstream::badbit);
    try {
        in.open(rule_file.c_str(), ifstream::in);
    }
    catch (system_error& e) {
	cout << "Error: failed to open rule file." << endl;
	exit(1);
    }
    for (int i = 0; i < RULELENGTH; ++i) {
	in >> x;
	if (in.eof())
	    throw IncorrectRuleLengthException();
	_rule[i] = x - 48;
    }
    //in >> x;
    //if (in.eof())
    //	throw IncorrectRuleLengthException();
    in.close();
}


/***************************************************************************************/

// Load two CA rules from rule_file for non_uniform reservoir
// from successive lines of rule_file where first line ends with ':' 
void CA::load_two_rules(string rule_file) {
    ifstream in;
    char x;
    int count = 0;

    in.exceptions(ifstream::failbit | ifstream::badbit);
    try {
        in.open(rule_file.c_str(), ifstream::in);
    }
    catch (system_error& e) {
	cout << "Error: failed to open rule file." << endl;
	exit(1);
    }
    for (int i = 0; i < RULELENGTH; ++i) {
	in >> x;
	if (in.eof())
	    throw IncorrectRuleLengthException();
	_rule[i] = x - 48;
    }
    while (x != ':') {
	++count;
        if (count == 100)
	    throw NonUniformRuleFileFormatException();
        in >> x;
    }
    for (int i = 0; i < RULELENGTH2; ++i) {
	in >> x;
	if (in.eof())
	    throw IncorrectRuleLengthException();
	_rule2[i] = x - 48;
    }
    in.close();
}

/***************************************************************************************/

// Apply inputs to _iter = 0 row of CA
void CA::set_input(vector<int> input) {
    int i, j;

    _iter = 0;
    for (i = 0; i < R; ++i) {
	for (j = 0; j < INPUT_LENGTH; ++j) {
	    // Overwrite initial row with mapped inputs
            _cell[0][i * DIFFUSE_LENGTH + _map[i][j]] = input[j];
	    // Add input state + 1 to initial row
            //_cell[0][i * DIFFUSE_LENGTH + _map[i][j]] = 
		//(_cell[0][i * DIFFUSE_LENGTH + _map[i][j]] + input[j] + 1) % STATES;
	}
    }
}

/***************************************************************************************/

// Apply inputs to _iter = 0 row of CA for temporal parity task
void CA::set_parity_input(vector<int> input) {
    int i, j;

    _iter = 0;
    for (i = 0; i < R; ++i) {
        for (j = 0; j < PARITY_INPUT_LENGTH; ++j) {
            // Overwrite initial row with mapped inputs
            if (OV)
                _cell[0][i * DIFFUSE_LENGTH + _map[i][j]] = input[j];
            // Add input state + 1 to initial row
            else
                _cell[0][i * DIFFUSE_LENGTH + _map[i][j]] = 
                    (_cell[0][i * DIFFUSE_LENGTH + _map[i][j]] + input[j] + 1) % STATES;
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

// Build CA without applying inputs
void CA::check_CA(real_2d_array& training_data, bool uniform) {
    int data_index = 0;

    for (int i = 0; i < READOUT_LENGTH; ++i) {
        if (uniform)
	   apply_rule(training_data, data_index++);
        else
            apply_two_rules(training_data, data_index++);
    }

    //save_CA(training_data);
    //draw_CA(training_data);
}


/***************************************************************************************/

// Generate ca.ppm file of first WIDTH rows of CA in ca.txt
void CA::draw_CA(alglib::real_2d_array& training_data, int height) {
    int i, j, k, l;
    char ans;
    int num_colors = 3 * max(STATES, STATES2);
    char charState, state;
    int layer[WIDTH], image_length;
    //int height = SEQUENCE_LENGTH * TEST_SETS;
    do { 
	FILE* f_out = fopen("ca.ppm", "w"); 
	FILE* f_in = fopen("ca.txt", "r");

	fputs("P3\n", f_out);
	// Square PPM image of beginning of training data
        if (!LONG_DRAW)
            fprintf(f_out, "%d %d\n", 3 * WIDTH, 3 * WIDTH);
        // For longer drawing that shows different test set inputs
        else 
            fprintf(f_out, "%d %d\n", 3 * WIDTH, 30 * WIDTH);
        fputs("255\n", f_out);
	vector<int> colors(num_colors);
	// Set colors randomly
        for (i = 0; i < num_colors; ++i) {
           if (i < 3) {
              colors[i] = 255;
           }
           else if (i < 6) {
              colors[i] = 0;
           }
           else if (i < 9) {
              colors[6] = 0;
              colors[7] = 0;
              colors[8] = 255;
           }
           else if (i < 12) {
              colors[9] = 255;
              colors[10] = 0;
              colors[11] = 0;
           }
           else 
              colors[i] = rand() % 256;
        }
        if (!LONG_DRAW)
            image_length = WIDTH;
        else
            image_length = 20*WIDTH;
        //for (i = 0; i < WIDTH; ++i)
        //for (i = 0; i < 10*WIDTH; ++i)   // longer drawing
        for (i = 0; i < image_length; ++i)
        {
       	  for (j = 0; j < WIDTH; ++j)
	  {
	     ans = (int)fscanf(f_in, " %c", &charState);
	     state =  (int) charState - 48;
	     layer[j] = state;
	     // Draw cells as 3 x 3 blocks
	     for (k = 0; k < 3; ++k)
		fprintf(f_out, "%d %d %d ", colors[state*3], colors[state*3+1], colors[state*3+2]);
	     if (i % 3 == 2)
		fprintf(f_out, "\n");
	  }
	  for (j = 0; j < 2; ++j)
	   for (l = 0; l < WIDTH; ++l)
	      for (k = 0; k < 3; ++k)
		 fprintf(f_out, "%d %d %d ", colors[layer[l]*3], colors[layer[l]*3+1], colors[layer[l]*3+2]);
       }
       fclose(f_out);
       fclose(f_in);
       cout << "Redraw CA? (y or n)  ";
       cin >> ans;
    } while (ans == 'y' || ans == 'Y');
}


/***************************************************************************************/

// Apply CA rule for I iterations.
// Copy last row to initial position ready to receive input data
// Changed to write input into training data
void CA::apply_rule(real_2d_array& training_data, int data_index) {
    int i, j;
    int rule_index, rule_window[NEIGHBORHOOD];
    int start = -(NEIGHBORHOOD / 2);
    int finish = NEIGHBORHOOD / 2;
    int n = RULELENGTH - 1;
    int neighbor_count;

    _iter = 0;
    //cout << "Applying rule\n";
    for (; _iter < I; ++_iter) {
	for (i = 0; i < WIDTH; ++i) {
	    if (DENSITY_RULE) {
		neighbor_count = 0;
		for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index) {
		    if (_cell[_iter][mod(i + j, WIDTH)] == 1) {
			if (j != 0)
			    ++neighbor_count; 
		    }
		    _cell[_iter + 1][i] = _rule[_cell[_iter][i] * NEIGHBORHOOD + neighbor_count];
		}
	        // Apply rule
	    }
	    else {
		for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index)
		    rule_window[rule_index] = _cell[_iter][mod(i + j, WIDTH)];
	        _cell[_iter + 1][i] = _rule[n -  base_N_to_dec(rule_window, STATES, NEIGHBORHOOD)];
	    }
	    training_data[data_index][i + WIDTH * _iter] = _cell[_iter][i];
	}
    }
    // copy last row to initial position
    for (i = 0; i < WIDTH; ++i)
	_cell[0][i] = _cell[_iter][i];
    //cout << "Data_index: " << data_index << endl;
}
/*
void CA::apply_rule(real_2d_array& training_data, int data_index) {
    int i, j;
    int rule_index, rule_window[NEIGHBORHOOD];
    int start = -(NEIGHBORHOOD / 2);
    int finish = NEIGHBORHOOD / 2;
    int n = RULELENGTH - 1;

    _iter = 0;
    //cout << "Applying rule\n";
    for (; _iter < I; ++_iter) {
	for (i = 0; i < WIDTH; ++i) {
	    for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index)
		rule_window[rule_index] = _cell[_iter][mod(i + j, WIDTH)];
	    _cell[_iter + 1][i] = training_data[data_index][i + WIDTH * _iter]
		= _rule[n -  base_N_to_dec(rule_window, STATES, NEIGHBORHOOD)];
	}
    }
    // copy last row to initial position
    for (i = 0; i < WIDTH; ++i)
	_cell[0][i] = _cell[_iter][i];
    //cout << "Data_index: " << data_index << endl;
}
*/
/***************************************************************************************/

// Apply two CA rule for I iterations for non-uniform reservoir..
// Copy last row to initial position ready to receive input data
void CA::apply_two_rules(real_2d_array& training_data, int data_index) {
    int i, j;
    int rule_switch = WIDTH / 2;
    int rule_index, rule_window[NEIGHBORHOOD];
    int start = -(NEIGHBORHOOD / 2);
    int finish = NEIGHBORHOOD / 2;
    int n = RULELENGTH - 1;
    int neighbor_count;
    // For NU_RULES_SAME_TYPE = false
    int rule_window2[NEIGHBORHOOD2];
    int start2 = -(NEIGHBORHOOD2 / 2);
    int finish2 = NEIGHBORHOOD2/ 2;
    int n2 = RULELENGTH2 - 1;


    _iter = 0;
    //cout << "Applying rule\n";
    for (; _iter < I; ++_iter) {
	for (i = 0; i < WIDTH; ++i) {
	    if (DENSITY_RULE) { // NU_RULES_SAME_TYPE == true not defined for DENSITY_RULE
		neighbor_count = 0;
		for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index) {
		    if (_cell[_iter][mod(i + j, WIDTH)] == 1) {
			if (j != 0)
			    ++neighbor_count; 
		    }
		    if (i < rule_switch)
			_cell[_iter + 1][i] = _rule[_cell[_iter][i] * NEIGHBORHOOD + neighbor_count];
		    else
			_cell[_iter + 1][i] = _rule2[_cell[_iter][i] * NEIGHBORHOOD + neighbor_count];
		}
	    }
	    else {
	        if (i < rule_switch) { // rule 1
		   for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index)
		       rule_window[rule_index] = mod(_cell[_iter][mod(i + j, WIDTH)], STATES);
		   _cell[_iter + 1][i] = _rule[n -  base_N_to_dec(rule_window, STATES, NEIGHBORHOOD)];
		}
		else if (NU_RULES_SAME_TYPE) { // rule 2
		   for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index)
		       rule_window[rule_index] = _cell[_iter][mod(i + j, WIDTH)];
		   _cell[_iter + 1][i] = _rule2[n -  base_N_to_dec(rule_window, STATES, NEIGHBORHOOD)];
		}
		else if (!RULE_2_POP_DENS) {
		   for (j = start2, rule_index = 0; j <= finish2; ++j, ++rule_index)
		       rule_window2[rule_index] = mod(_cell[_iter][mod(i + j, WIDTH)], STATES2);
		   _cell[_iter + 1][i] = _rule2[n2 -  base_N_to_dec(rule_window2, STATES2, NEIGHBORHOOD2)];
		}
		else {
		    neighbor_count = 0;
		    for (j = start2, rule_index = 0; j <= finish2; ++j, ++rule_index)
		    if (_cell[_iter][mod(i + j, WIDTH)] == 1) {
			if (j != 0)
			    ++neighbor_count; 
		    }
		    _cell[_iter + 1][i] = _rule2[_cell[_iter][i] * NEIGHBORHOOD2 + neighbor_count];
		}
	    }
	    training_data[data_index][i + WIDTH * _iter] = _cell[_iter][i];
	}
    }
    // copy last row to initial position
    for (i = 0; i < WIDTH; ++i)
	_cell[0][i] = _cell[_iter][i];
    //cout << "Data_index: " << data_index << endl;
}

/***************************************************************************************/

// Generate targets for 5 bit memory task
void CA::set_5_bit_targets() {
    int data_index, i, test_set, time_step;
    int distractor_end = SEQUENCE_LENGTH - 5;

    data_index = 0;
    for (test_set = 0; test_set < TEST_SETS; ++test_set) {
	for (i = 0; i < distractor_end; ++i) {
	    _targets[0][data_index] = 0;
            _targets[1][data_index] = 0;
            _targets[2][data_index] = 1;
	    ++data_index;
	}
	// Recall period
	for (time_step = 0; i < SEQUENCE_LENGTH; ++i, ++time_step) {
	    _targets[0][data_index] = test_set >> time_step & 1;
            _targets[1][data_index] = 1 - (test_set >> time_step & 1);
	    _targets[2][data_index] = 0;
	    ++data_index;
	}
    }
}

/***************************************************************************************/
   
// Train and test reservoir using SVMTorch support vector machines
void CA::call_SVM_functions(int model, int& incorrect, real_2d_array training_data) {
    string build_model = "./SVMTorch  SVM"; 
    string test_results = "./SVMTest -oa SVM_results"; 
    string output_file = "SVM_results";
    string data_file = "SVM";
    int SVMtag;
    int tid = omp_get_thread_num();
    int system_result;
    ofstream out;
    ifstream in;

    out.open((data_file + to_string(tid) +".dat").c_str(), ofstream::out);
    out << SEQUENCE_LENGTH * TEST_SETS << " " << READOUT_LENGTH + 1 << endl;
    // Build input file for SVMTorch
    for (int i = 0; i < SEQUENCE_LENGTH * TEST_SETS; ++i) {
	for (int j = 0; j < READOUT_LENGTH; ++j) {
	    out << training_data[i][j] << " ";
	}
	SVMtag = _targets[model][i] == 1 ? 1 : -1; 
	out << SVMtag << endl;
    }
    // Build and test model
    out.close();
    system_result = system((build_model + to_string(tid) + ".dat SVM_model" + to_string(tid)).c_str());
    puts((build_model + to_string(tid) + ".dat SVM_model" + to_string(tid)).c_str());
    system_result = system((test_results + to_string(tid) + ".dat SVM_model" + 
	    	to_string(tid) + " SVM" + to_string(tid) + ".dat").c_str());
    puts((test_results + to_string(tid) + ".dat SVM_model" +
		to_string(tid) + " SVM" + to_string(tid) + ".dat").c_str());
    in.open((output_file + to_string(tid) + ".dat").c_str(), ifstream::in);
    float result;
    for (int i = 0; i < SEQUENCE_LENGTH * TEST_SETS; ++i) {
	in >> result;
	if ((result < 0 && _targets[model][i] == 1) || (result >= 0 && _targets[model][i] == 0)) { 
	    #pragma omp critical
	    {
		++incorrect;
	    }
	}
    }
    in.close();
}

/***************************************************************************************/

// Nested parallelism wasn't working. Could it cause tid #s to be reused?
int CA::build_SVM_model(real_2d_array& training_data) {
    int incorrect = 0;

//    #pragma omp parallel sections
//    {
//	#pragma omp section
//	{   // model 0
	    call_SVM_functions(0, incorrect, training_data);
//	}
 //      #pragma omp section
//	{  // model 1
	    call_SVM_functions(1, incorrect, training_data);
//	}
//	#pragma omp section
//	{ // model 2
	    call_SVM_functions(2, incorrect, training_data);
//	}
 //   }
    cout << "\nIncorrect: " << incorrect << endl;
    return incorrect;
}

/***************************************************************************************/

// Build and test linear regression models using python scikit library via
// helper script build_model.py
void CA::python_regression(int model, int& incorrect, real_2d_array training_data) {
    string build_model = "python3 build_model.py sk";
    string data_file = "sk";
    string output_file =  "sk_results";
    //string tags_file = "sk_tags";
    int tid = omp_get_thread_num();
    int sk_tag;
    int system_result;
    ofstream out;
    ifstream in;

    out.open((data_file + to_string(tid) + ".csv").c_str(), ofstream::out);
    for (int i = 0; i < SEQUENCE_LENGTH * TEST_SETS; ++i) {
        for (int j = 0; j < READOUT_LENGTH - 1; ++j) {
	    out << training_data[i][j] << " ";
	}
	// Last data w/o a space
	out << training_data[i][READOUT_LENGTH - 1] << endl;
    }
    out.close();
    
    // This file of tags does not change and only needs built once
    /*
    out.open((tags_file + to_string(model) + ".txt").c_str(), ofstream::out);
    for (int i = 0; i < SEQUENCE_LENGTH * TEST_SETS - 1; ++i) {
	//sk_tag = _targets[model][i] == 1 ? 1 : -1;
	out << _targets[model][i] << " ";
    }
    //sk_tag = _targets[model][SEQUENCE_LENGTH * TEST_SETS - 1] == 1 ? 1 : -1;
    out << _targets[model][SEQUENCE_LENGTH * TEST_SETS - 1];
    out.close();
    */
   
    system_result = system((build_model + to_string(tid) + ".csv sk_tags" + to_string(model)
		+ ".txt sk_results" + to_string(tid) + ".csv").c_str());
    puts((build_model + to_string(tid) + ".csv sk_tags" + to_string(model) + ".txt sk_results"
		+ to_string(tid) + ".csv").c_str());
    in.open((output_file + to_string(tid) + ".csv").c_str(), ifstream::in);
    float result;
    for (int i = 0; i < SEQUENCE_LENGTH * TEST_SETS; ++i) {
	in >> result;
	if ((result < .5 && _targets[model][i] == 1) || (result >= .5 && _targets[model][i] == 0)) {
            #pragma omp critical
	    {
       	        ++incorrect;
		cout << "Model: " << model+1 << "\tTest Set: " << i/SEQUENCE_LENGTH << 
		    "\tSequence #: " << i%SEQUENCE_LENGTH  << "\tCalcuated: " << result <<
		    "\tTarget: " << _targets[model][i] << endl;
	    }
	}
    }
    in.close();
}

/***************************************************************************************/

// Use python scikit to build and test three linear regression rules
int CA::build_scikit_model(real_2d_array& training_data) {
    int incorrect = 0;
    clock_t t;

    if (TIME)
       t = clock();
    python_regression(0, incorrect, training_data);
    // Don't continue testing if there are errors if (incorrect < 1)
    if (TIME || incorrect < 1)
        python_regression(1, incorrect, training_data);
    if (TIME || incorrect < 1)
        python_regression(2, incorrect, training_data);
    cout << "\nIncorrect: " << incorrect << endl;
    if (TIME) {
       t = clock() - t;
       cout << "Elapsed CPU time: " << (float)t/CLOCKS_PER_SEC << " seconds\n";
    }
    return incorrect;
}

/***************************************************************************************/

// Create AlgLib linear regression models for 5 bit task
void CA::build_5_bit_model(real_2d_array& training_data, vector<linearmodel>& output) {
    int time_step1, test_set1, data_index1, data_index2, data_index3;
    int time_step2, test_set2, test_set3;
    int i, j, k, stop = SEQUENCE_LENGTH * TEST_SETS; 
    int distractor_end = SEQUENCE_LENGTH - 5;
    //int model_index;
    ae_int_t info;
    //ae_int_t nvars;   // for lrunpack()
    lrreport rep;

    // Copy data so regressions can be performed in parallel
    real_2d_array training_data2;
    real_2d_array training_data3;
    training_data2.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);
    training_data3.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH + 1);
    for (i = 0; i < stop; ++i) {
	for (j = 0; j < READOUT_LENGTH; ++j) {
	    training_data3[i][j] = training_data2[i][j] = training_data[i][j];
	}
    }

    #pragma omp parallel sections
    {
        #pragma omp section
	{   // model 0	
	    data_index1 = 0;
	    for (test_set1 = 0; test_set1 < TEST_SETS; ++test_set1) {
		for (i = 0; i < distractor_end; ++i) {
		    //_targets[0][data_index1] = 0;
		    training_data[data_index1][READOUT_LENGTH] = 0;
		    ++data_index1;
		}
		// Recall period
		for (time_step1 = 0; i < SEQUENCE_LENGTH; ++i, ++time_step1) {
		    training_data[data_index1][READOUT_LENGTH] = 
			test_set1 >> time_step1 & 1;
		    //_targets[0][data_index1] = training_data[data_index1][READOUT_LENGTH];
		    ++data_index1;
		}
	    }
	    cout << "Building linear regression model #1\n";
	    lrbuildz(training_data, SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH, info,
		    output[0], rep);    // Try lrbuildz()
	}
        #pragma omp section
	{  // model 1
	    data_index2 = 0;
	    for (test_set2 = 0; test_set2 < TEST_SETS; ++test_set2) {
		for (j = 0; j < distractor_end; ++j) {
		    //_targets[1][data_index2] = 0;
		    training_data2[data_index2][READOUT_LENGTH] = 0;
		    ++data_index2;
		}
		// Recall period
		for (time_step2 = 0; j < SEQUENCE_LENGTH; ++j, ++time_step2) {
		    training_data2[data_index2][READOUT_LENGTH] = 
			1 - (test_set2 >> time_step2 & 1);
		    //_targets[1][data_index2] = training_data2[data_index2][READOUT_LENGTH];
		    ++data_index2;
		}
	    }
	    cout << "Building linear regression model #2\n";
	    lrbuildz(training_data2, SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH, info,
		    output[1], rep);    // Try lrbuildz()
	}
        #pragma omp section
	{ // model 2
	    data_index3 = 0;
	    for (test_set3 = 0; test_set3 < TEST_SETS; ++test_set3) {
		for (k = 0; k < distractor_end; ++k) {
		    //_targets[2][data_index3] = 1;
		    training_data3[data_index3][READOUT_LENGTH] = 1;
		    ++data_index3;
		}
		// Recall period
		for (; k < SEQUENCE_LENGTH; ++k) {
		    //_targets[2][data_index3] = 0;
		    training_data3[data_index3][READOUT_LENGTH] = 0;
		    ++data_index3;
		}
	    }
	    cout << "Building linear regression model #3\n";
	    lrbuildz(training_data3, SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH, info,
		    output[2], rep);    // Try lrbuildz()
	}
    }


    /*for (i = 0; i < 3; ++i) {     // Print out coefficients
	real_1d_array coeffs;
	lrunpack(output[i], coeffs, nvars);
	printf("Coefficients: %s\n", coeffs.tostring(4).c_str());
    }*/

    // Print out targets
    /*cout << "Output\n";
    for (i = 0; i < 32; ++i) {
	cout << "Test set # " << i << endl;
	for (int j = 0; j < 5; ++j) {
	    cout << "\t" << _targets[0][i*210 +  205+j] << " " <<_targets[1][i*210 +  205+j] <<
                " " << _targets[2][i*210 +  205+j] << endl;
	}
    }*/



}

/***************************************************************************************/
 
// Test CA reservoir for 5 bit task using AlgLib linear regression models
int CA::test_5_bit(real_2d_array& training_data, vector<linearmodel>& output) {
    real_1d_array model_input;
    int incorrect_predictions = 0;
    int model_index = 0;
    int training_data_index, test_set, sequence_index, i;
    double result;
    int result_state;

    cout << setprecision(4);
    model_input.setlength(READOUT_LENGTH);
    for (model_index = 0; model_index < 3; ++model_index) {
	training_data_index = 0;
        for (test_set = 0; test_set < TEST_SETS; ++test_set) {
	    for (sequence_index = 0; sequence_index < SEQUENCE_LENGTH; ++sequence_index) {
		// Copy reservoir sequence into model_input
		for (i = 0; i < READOUT_LENGTH; ++i)
		    model_input[i] = training_data[training_data_index][i];
		result = lrprocess(output[model_index], model_input);
		result_state = result < .5 ? 0 : 1;
                if (result_state != _targets[model_index][training_data_index]) {
		    ++incorrect_predictions;
		    cout << "Model: " << model_index+1 << "\tTest Set: " 
			<< test_set << "\tSequence #: " << sequence_index + 1 << 
			"\tCalcuated: " << result << "\tTarget: " <<
			 _targets[model_index][training_data_index] << endl;
		}
		//else cout << "1    ";
		//cout << result << "\t\t" << training_data[training_data_index][READOUT_LENGTH]
		//    << endl;
		//cout << result << "\t\t" << _targets[model_index][training_data_index] << endl;
		++training_data_index;
	    }
	}
    }
    cout << endl << incorrect_predictions << " incorrect predictions.\n";
    return incorrect_predictions;
}

/***************************************************************************************/

// Save and draw CA reservoir
void CA::save_CA(real_2d_array& training_data, int height) {
    int i, j;
    //int height = SEQUENCE_LENGTH * TEST_SETS;
    int width = READOUT_LENGTH;
    FILE* f_out = fopen("ca.txt", "w");
    
    for (i = 0; i < height; ++i) {
	for (j = 0; j < width; ++j)
	    fprintf(f_out, "%d", (int)training_data[i][j]);
    }
    fclose(f_out);

    draw_CA(training_data, height);
}

/***************************************************************************************/

// Build CA with 5 bit task input
void CA::train_5_bit(real_2d_array& training_data, bool uniform) {
    int time_step ,test_set; 
    int data_index = 0;
    int distractor_end = SEQUENCE_LENGTH - 6;
    vector<int> input(4);

    //cout << "Input\n";
    for (test_set = 0; test_set < TEST_SETS; ++test_set) {
	//cout << "Test Set: " << test_set << endl;
	// Input signal
	for (time_step = 0; time_step < 5; ++time_step) {
	    input[0] = test_set >> time_step & 1;
	    input[1] = !input[0];
	    input[2] = input[3] = 0;
	    set_input(input);
	    if (uniform)
                apply_rule(training_data, data_index++); 
	    else
                apply_two_rules(training_data, data_index++); 
	    //cout << "\t" << input[0] << " " << input[1] << " " << input[2] << " " << input[3] << endl;
	}
	// Distractor period
	for (; time_step < distractor_end; ++time_step) {
	    input[0] = input[1] = input[3] = 0;
	    input[2] = 1;
	    set_input(input);
	    if (uniform)
                apply_rule(training_data, data_index++); 
	    else
                apply_two_rules(training_data, data_index++); 
	}
	// Distractor signal
	input[0] = input[1] = input[2] = 0;
	input[3] = 1;
	set_input(input);
	if (uniform)
	    apply_rule(training_data, data_index++); 
	else
	    apply_two_rules(training_data, data_index++); 
	++time_step;
        // Recall period
	for (; time_step < SEQUENCE_LENGTH; ++time_step) {
	    input[0] = input[1] = input[3] = 0;
	    input[2] = 1;
	    set_input(input);
	    if (uniform)
                apply_rule(training_data, data_index++); 
	    else
                apply_two_rules(training_data, data_index++); 
	}
    }
    //cout << "Data_index: " << data_index << endl;
}

/***************************************************************************************/

// Generate a random rule
void random_rule(vector<int>& rule) {
    for (int i = 0; i < RULELENGTH; ++i) 
	rule[i] = rand() % STATES;
}

/***************************************************************************************/

void stochastic_3_state_temp_dens(int runs) {
    //cout << "Stochastic search for 3 state temporal density task rules\n";
    string output_file;
    if (STATES == 3)
       output_file = "temp_dens.txt";
    else if (NEIGHBORHOOD == 5)
       output_file = "temp_dens_nei_5.txt";
    ofstream out;
    int good_CA_count = 0;

    out.open(output_file, ofstream::out | ofstream::app);
    vector<int> input = {0, 1, 0, 1};
    int i, j , epoch, data_index = 0;
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (i = 0; i < runs; ++i) {
            int scores[2];
            string rule;
            random_rule_string(rule);
            temporal_density(true, false, "-1", scores, rule, "");
            cout << "Scores: "  << scores[0] << "\t" << scores[1] << endl;
            if (scores[0] == 0 || scores[1] == 0) {
                #pragma omp critical
                {
                    ++good_CA_count;
                    out << rule << " " << scores[0] << "/" << scores[1] << endl;
                }
            }
        }
    } 
    cout << "Good CAs: " << good_CA_count << endl;
    out.close();
}

/***************************************************************************************/

void random_rule_string(string& rule) {
    for (int i = 0; i < RULELENGTH; ++i)
      rule = rule + to_string(rand() % STATES);
}    

/***************************************************************************************/

// Evolve and test four and five state rules 

void  build_4_state_file(int runs) {
    string syst = "./CA ev_";
    string rule_file = "ev_";
    string output_file;
    //string output_file = "evolved_rules.txt";
    //string output_file = "neighborhood_7_rules.txt";
    int good_ca_count = 0;
    ofstream out;

    if (TD) {
        if (STATES == 4)
            output_file = "temp_dens_four_state.txt";
        else if (STATES == 5)
            output_file = "temp_dens_five_state.txt";
    }
    else {
        if (STATES == 4)
            output_file = "four_state_rules_smallest_largest.txt";
        else if (STATES == 5)
            output_file = "five_state_rules.txt";
    }
    out.open(output_file, ofstream::out | ofstream::app); 
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < runs; ++i) {
            srand(time(NULL) + omp_get_thread_num());
            this_thread::sleep_for(chrono::milliseconds(1000 * 
                        omp_get_thread_num()));
            int incorrect = 0;
            int tid = omp_get_thread_num();
            system((syst + to_string(tid)).c_str());
            if (TD) {
                int scores[2];
                temporal_density(true, false, ("ev_" + to_string(tid)).c_str(), scores, "-1", "-1");
                cout << "Scores: "  << scores[0] << "\t" << scores[1] << endl;
                if (scores[0] < 5 || scores[1] < 5) {
                    #pragma omp critical
                    {
                        ifstream in;
                        in.open(("ev_" + to_string(tid)).c_str(), ifstream::in);
                        ++good_ca_count;
                        char n;
                        for (int j = 0; j < RULELENGTH; ++j) {
                            in >> n;
                            out << n;
                            //out << ca._rule[j];
                        }
                        out << " " << scores[0] << "/" << scores[1] << endl;
                    }
                }
            }
            else {
                CA ca(false, 0);
                ca.load_rule(("ev_" + to_string(tid)).c_str());
                real_2d_array training_data;
                training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH);
                ca.train_5_bit(training_data, true);
                ca.set_5_bit_targets();
                incorrect = ca.build_scikit_model(training_data);
                
                if (incorrect < 1) {
                    #pragma omp critical
                    {
                        ++good_ca_count;
                        for (int j = 0; j < RULELENGTH; ++j)
                            out << ca._rule[j];
                        out << " " << incorrect << "\n";
                    }
                }
            }
            // Draw single CA
            //ca.save_CA(training_data, SEQUENCE_LENGTH * TEST_SETS);
        }
    }
    cout << "Good CAs: " << good_ca_count << endl;
    out.close();
} 

/***************************************************************************************/
/*
void build_3_state_temp_dens_file(int runs, bool uniform) {
   string output_file = "temp_dens.txt";
   ofstream out;
   ifstream in;
   string rule, rule_name, rule2_name;
   string rule2 = "-1";
   int scores[2];
   char junk[100];

   try {
       in.open(input_file.c_str(), ifstream::in);
   }
   catch (system_error& e) {
       cout << "Error: failed to open rule file." << endl;
       exit(1);
   }
   out.open(output_file.c_str(), ofstream::out | ofstream::app);
   while (in >> rule) {
       in >> junk;
       in >> rule_name;
       cout << rule << "  " << rule_name << endl;;
       in.getline(junk, 100);
       if (!uniform) {
           in >> rule2;
           cout << rule2 << "  ";
           in >> junk;
           in >> rule2_name;
           cout << rule2_name << endl;
       } 
       temporal_density(uniform, false, "-1", scores, rule, rule2);
       cout << scores[0] << "    " << scores[1] << endl;
   }
}
*/
/***************************************************************************************/

// Stochastic search for promising neighborhood 5 rules.
// Eliminate Class 1 and 2 rules and append rules
// with low error on 5 bit task
// to neighborhood_5_rules.txt
void build_5_neighborhood_file(int runs, bool uniform, bool scikit) {
    int good_CA_count = 0;
    int reject_CAs    = 0;
    ofstream out;

    if (TD)
        out.open("temp_dens_nei_5.txt", ofstream::out | ofstream::app);
    else
        out.open("neighborhood_5_rules.txt", ofstream::out | ofstream::app);
    //out.open("neighborhood_7_rules.txt", ofstream::out | ofstream::app);
    try {
	if (STATES != 2 || NEIGHBORHOOD != 5) throw Expect2State5NeighborhoodException();
	//if (STATES != 2 || NEIGHBORHOOD != 7) throw Expect2State5NeighborhoodException();
    }
    catch(Expect2State5NeighborhoodException e)
    {
	cout << "Error: STATES must be 2 and NEIGHBORHOOD 5 for -bn option.\n"; 
	exit(1);
    }
     vector<int> input = {0,1,0,1};
    #pragma omp parallel
    {
	#pragma omp for nowait
	for (int i = 0; i < runs; ++i) {
            vector<int> rule(RULELENGTH);
	    random_rule(rule);
	    int errors;
	    CA ca;
	    // Handle this case in stochastic_3_state_temp_dens
	    if (TD) {
	       string rule1;
	    }
	    else {
	       real_2d_array training_data;
	       vector<linearmodel> output(3);
	       //training_data.setlength(READOUT_LENGTH, READOUT_LENGTH);
	       //We need extra size for save -- remove after testing
	       training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH);
	       ca.set_rule(rule);
	       ca.set_input(input);
	       /*for (j = 0; j < 32; ++j) 
		   cout<< rule[j];
	       cout << endl;*/
	       //if (i % 1000 == 0)
	       //		cout << i << endl;
	       ca.check_CA(training_data, uniform);
	       if (!find_static_CAs(training_data)) {
		   //ca.save_CA(training_data);
		   ca.train_5_bit(training_data, uniform);
		   if (scikit) {
		       ca.set_5_bit_targets();
		       errors = ca.build_scikit_model(training_data);
		   }
		   else {
		       ca.build_5_bit_model(training_data, output);
		       errors = ca.test_5_bit(training_data, output);
		   }
		   if (errors < 10) {
		   #pragma omp critical
		       {
			   ++good_CA_count;
			   for (int j = 0; j < RULELENGTH; ++j)
			       out << rule[j];
			   out << " " << errors << "\n";
		       }
		   }
	       }
	       else {
	       #pragma omp critical
		   {
		       ++reject_CAs;
		   }
	       }
	   }
	}
    }
    cout << "Good CAs: " << good_CA_count << endl;
    cout << "Rejected as static: "  << reject_CAs << "\n";
    out.close();
}   

/***************************************************************************************/

// Stochastic search for promising rules.
// Eliminate Class 1 and 2 rules and append rules
// with low error on 5 bit task
// to three_state_rules.txt
void build_3_state_CA_file(int runs, bool uniform, bool scikit) {
    int good_CA_count = 0;
    int reject_CAs    = 0;
    ofstream out;

    out.open("three_state_rules.txt", ofstream::out | ofstream::app);
    //out.open("four_state_rules.txt", ofstream::out | ofstream::app);
    try {
	if (I < 3) throw BuildRuleFileRequiresIAtLeast3Exception();
    }
    catch(BuildRuleFileRequiresIAtLeast3Exception e)
    {
	cout << "Error: 'I' must be at least three for -bf option.\n"; 
	exit(1);
    }
    vector<int> input = {0,1,2,1};
    int i, j, epoch, data_index = 0;
    #pragma omp parallel
    {
	#pragma omp for nowait
	for (i = 0; i < runs; ++i) {
	    CA ca;
            vector<int> rule(RULELENGTH);
	    int errors;
	    real_2d_array training_data;
	    vector<linearmodel> output(3);
	    //training_data.setlength(READOUT_LENGTH, READOUT_LENGTH);
	    //We need extra size for save -- remove after testing
	    training_data.setlength(SEQUENCE_LENGTH * TEST_SETS, READOUT_LENGTH);
	    random_rule(rule);
	    ca.set_rule(rule);
	    ca.set_input(input);
	    /*for (j = 0; j < 27; ++j) 
		cout<< rule[j];
	    cout << endl;*/
	    //if (i % 1000 == 0)
            //		cout << i << endl;
	    ca.check_CA(training_data, uniform);
	    if (!find_static_CAs(training_data)) {
                //ca.save_CA(training_data);
		ca.train_5_bit(training_data, uniform);
		if (scikit) {
		    ca.set_5_bit_targets();
		    errors = ca.build_scikit_model(training_data);
		}
		else {
		    ca.build_5_bit_model(training_data, output);
		    errors = ca.test_5_bit(training_data, output);
		}
		if (errors < 1) {
		#pragma omp critical
		    {
			++good_CA_count;
			for (j = 0; j < RULELENGTH; ++j)
			    out << rule[j];
			out << " " << errors << "\n";
		    }
		}
	    }
	    else {
            #pragma omp critical
		{
                    ++reject_CAs;
		}
            }
	}
    }
    cout << "Good CAs: " << good_CA_count << endl;
    cout << "Rejected as static: "  << reject_CAs << "\n";
    out.close();
}

/***************************************************************************************/

// Identify Class 1 and Class 2 convergent rules
bool find_static_CAs(real_2d_array& training_data) {
    bool flag;
    int index1, index2;

    // Check if last row matches either of 2 preceding
    flag = true;
    for (index1 = 0, index2 = WIDTH; index1 < WIDTH; ++index1, ++index2) {
	//if (training_data[READOUT_LENGTH-1][index1] != training_data[READOUT_LENGTH-1][index2]) {
        // Changed to check at 100th generation rather than last
	if (training_data[25][index1] != training_data[25][index2]) {
	    flag = false;
	    break;
	}
    }
    if (flag) {
	//cout << "last two match\n";
	return true;
    }
    flag = true;
    for (index1 = 0, index2 = 2*WIDTH; index1 < WIDTH; ++index1, ++index2) {
	if (training_data[25][index1] != training_data[25][index2]) {
	    flag = false;
	    break;
	}
    }
    if (flag) {
	//cout << "last matches 2nd to last\n";
	return true;
    }
    // Check if last row is shifted one cell right or left from previous row
    flag = true;
    for (index1 = 0, index2 = WIDTH+1; index1 < WIDTH-1; ++index1, ++index2) {
	if (training_data[25][index1] != training_data[25][index2]) {
	    flag = false;
	    break;
	}
    }
    if (flag) {
	//cout << "shifted right\n";
	return true;
    }
    flag = true;
    for (index1 = 1, index2 = WIDTH; index1 < WIDTH-1; ++index1, ++index2) {
	if (training_data[25][index1] != training_data[25][index2]) {
	    flag = false;
	    break;
	}
    }
    if (flag) {
	//cout << "shifted left\n";
	return true;
    }
    // Check if two rows above are shifted twice
    flag = true;
    for (index1 = 0, index2 = 2*WIDTH+1; index1 < WIDTH-2; ++index1, ++index2) {
	if (training_data[25][index1] != training_data[25][index2]) {
	    flag = false;
	    break;
	}
    }
    if (flag) {
	//cout << "shifted right in 2 levels\n";
	return true;
    }
    flag = true;
    for (index1 = 1, index2 = 2*WIDTH; index1 < WIDTH-2; ++index1, ++index2) {
	if (training_data[25][index1] != training_data[25][index2]) {
	    flag = false;
	    break;
	}
    }
    if (flag) {
	//cout << "shifted left in 2 levels\n";
	return true;
    }
    return false;
}

/***************************************************************************************/

void dec_to_base_3(vector<int>& result, int num) {
	for (int i = 26; i >= 0; --i) {
	    result[i] = num % 3;
	    num /= 3;
	}
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

/***************************************************************************************/

void CA::dec_to_base_N(int num, int base, vector<int>& ans) {
    int count = 0;
    while (num > 0) {
        ans[count++] = num % base;
	num /= base;
    }
}


/***************************************************************************************/

void dec_to_base_N(int num, int base, vector<int>& ans) {
    int count = 0;
    while (num > 0) {
        ans[count++] = num % base;
	num /= base;
    }
}


