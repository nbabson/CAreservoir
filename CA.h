// Neil Babson

#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <time.h>

using namespace std;

const int WIDTH = 400;  //400
const int GENERATIONS = 200; //200
const int STATES = 5;  //4
const int NEIGHBORHOOD = 3;   //3
const int POPULATION = 32;       //16   
const long int RULELENGTH = pow(STATES, NEIGHBORHOOD);
const int EPOCHS = 300; // 1000
const int TARGET_SIZE = 100000000;

struct CA
{
   int cell[WIDTH][GENERATIONS];
};

struct rule
{
   int r[RULELENGTH];
};

struct fit
{
   int f;
   int ca;
};
