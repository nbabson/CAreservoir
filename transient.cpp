
#include "transient.h"

int main() {
   srand(time(NULL));

   float rules_average = 0;
   //int num_rule = 59;
   //int num_rule = 32;
   int num_rule = 1;
   //string rule_file = "rule.txt";
   string rule_file = "rand_3_state.txt";
   //string rule_file = "four_state_rules_smallest_largest.txt";
   //string rule_file = "rand_4_state.txt";
   
   string transient_file = "rand_3_state_transients.txt";
   ifstream in;
   //ofstream out;

   in.open(rule_file.c_str(), ifstream::in); 
   //out.open(transient_file.c_str(), ofstream::out);

   //#pragma omp parallel
   {
      int i;
      int transient;  // disclosure length
      int trans;  // transient length
      int num_runs, total = 0;
      int trans_total = 0;
      int time_outs = 0;
      vector<int> r;
      char x;
      string junk;
      float average, trans_average;
      bool timed_out = false;

      r.resize(RULELENGTH);

      //save_random_rules();
      //exit(0);
      //#pragma omp for nowait
      for (int j = 0; j < num_rule; ++j) {
         //#pragma omp critical
         {
            for (int k = 0; k < RULELENGTH; ++k) {   // Save next rule in r
               in >> x;
               r[k] = x - 48;
            }
            getline(in, junk);
         }
         for (i = 0; i < RULELENGTH; ++i)
            cout << r[i] << " ";
         cout << endl;

         for (num_runs = 0; num_runs < RUNS; ++num_runs) {
            CA ca;
            ca.load_rule(r);
            
            if (timed_out) {
               timed_out = false;
               break;
            }
            for (i = 0; i < TIME_OUT; ++i) {
               ca.apply_rule();
               //ca.display_row();
               transient = ca.check_for_repeat(&trans);
               if (transient > 0) {
                  cout << "Transient length: " << trans << endl;
                  break;
               }
            }
            if (i == TIME_OUT) {
               cout << "TIMED OUT after " << TIME_OUT << " timesteps.\n";
               ++time_outs;
               total += TIME_OUT;
               trans_total += TIME_OUT;
               /*
               if (time_outs == 3) {
                  cout << "MAX TIMEOUTS = 3\n";
                  time_outs = 0;
                  timed_out = true;
                  continue;
                  //exit(0);
               }*/
            }
            else {
               total += transient;
               trans_total += trans;
            }
         }
         average = (float)total / (float)num_runs;
         cout << "Average disclosure length: " << average << endl;
         trans_average = (float)trans_total / (float)num_runs;
         cout << "Average transient length: " << trans_average << endl;
         //#pragma omp critical
         {
            rules_average += average;
            //out << average << endl;
         }
         total = 0;
      }
   }

   cout << "Average for all rules: " << rules_average / (float)num_rule << endl;
   in.close();
   //out.close();
   return 0;
}

/***************************************************************************************/

void load_next_rule(ifstream* in, vector<int>& r) {
   char x;
   string junk;
   for (int i = 0; i < RULELENGTH; ++i) {
      *in >> r[i];
   }
   getline(*in, junk);
}

/***************************************************************************************/

void CA::load_rule(vector<int> r) {
   for (int i = 0; i < RULELENGTH; ++i)
      rule[i] = r[i];
}

/***************************************************************************************/

CA::CA() {
   head = new row;
   tail = head;
   head -> data = new int[WIDTH];
   head -> next = NULL;
   random_init(head -> data);
   rule.resize(RULELENGTH);
   //load_rule("rule.txt");
   //load_rule("rand_3_state.txt");
   head -> length = 0;
   /*
   for (int i = 0; i < RULELENGTH; ++i)
      cout << rule[i] << " ";
   cout << endl;
   */
}

/***************************************************************************************/

CA::~CA() {
   while (head) {
      tail = head -> next;
      delete head -> data;
      delete head;
      head = tail;
   }
} 

/***************************************************************************************/

void CA::display_row() {
   for (int i = 0; i < WIDTH; ++i)
      cout << tail -> data[i] << " ";
   cout << endl;
}

/***************************************************************************************/

void random_init(int init[]) {
   for (int i = 0; i < WIDTH; ++i)
      init[i] = rand() % STATES;
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
    catch (exception& e) {
	cout << "Error: failed to open rule file." << endl;
	exit(1);
    }
    for (int i = 0; i < RULELENGTH; ++i) {
	in >> x;
	if (in.eof())
	    throw IncorrectRuleLengthException();
	rule[i] = x - 48;
    }
    in.close();
}


/***************************************************************************************/

void CA::apply_rule() {
    int i, j;
    int rule_index, rule_window[NEIGHBORHOOD];
    int start = -(NEIGHBORHOOD / 2);
    int finish = NEIGHBORHOOD / 2;
    int n = RULELENGTH - 1;
    int neighbor_count;
    int* new_data = new int[WIDTH];
    row* new_row = new row;

     for (i = 0; i < WIDTH; ++i) {
          for (j = start, rule_index = 0; j <= finish; ++j, ++rule_index)
              rule_window[rule_index] = tail -> data[mod(i + j, WIDTH)];
          new_data[i] = rule[n -  base_N_to_dec(rule_window, STATES, NEIGHBORHOOD)];
     }
     tail -> next = new_row;
     tail -> next -> length = tail -> length + 1;
     tail = new_row;
     tail -> data = new_data; 
     tail -> next = NULL;
}

/***************************************************************************************/

int CA::check_for_repeat(int* trans) {
    row* temp = head;
    bool match;
    while (temp != tail) {
       match = true;
       for (int i = 0; i < WIDTH; ++i) {
          if (temp -> data[i] != tail -> data[i])
             match = false;
       }
       if (match) {
          //cout << "Matches row " << temp -> length << endl;
          *trans = temp -> length;
          return tail -> length;
       }
       temp = temp -> next;
    }
    return -1;
}

/***************************************************************************************/

void save_random_rules() {
    ofstream out;
    int i, j;

    //out.open("rand_3_state.txt", ofstream::out | ofstream::app);
    out.open("rand_4_state.txt", ofstream::out | ofstream::app);
    for (i = 0; i < 31; ++i) {
       for (j = 0; j < RULELENGTH; ++j)
          out << rand() % STATES;
       out << endl;
    }
}

/***************************************************************************************/

int base_N_to_dec(int num[], int base, int length) {
    int total = 0;
    int place = 1;
    
    for (int i = length - 1; i >= 0; --i) {
	total += place * num[i];
        place *= base;
    }
    return total;
}

/***************************************************************************************/

int mod(int x, int y) {
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

