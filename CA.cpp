// Neil Babson
// March 20, 2016

#include "CA.h"
#include <iostream>

// Prototypes
void makeInitPop();
void evolAlg(int generation); 
void makeInitRand(int * initRand);
void makeCAs(rule * pop, CA * ca, int * initRand);
long int baseNtoDec(int * num, int base, int length);
void drawCA(CA ca, rule r);
void getFitness(fit * fitness);
int getArea(int x, int y, int state, int index);
float displayFitness(fit * fitness);
void sort(fit * fitness, int top, int bottom);
int partition(fit * fitness, int top, int bottom);
void mate(fit * fitness);
void mutate(fit * fitness);
void save(fit * fitness, int * best, float * average);

static rule pop[POPULATION];
static CA ca[POPULATION];
static bool visited[WIDTH][GENERATIONS];
string file_name;


int main(int argc, char* argv[])
{
   if (argc < 2) 
       file_name = "evolv_rule.txt";
   else 
       file_name = argv[1];
   srand(time(NULL));
   ifstream f_in;
   int args = 1, cell;
   /*if (argc > 1) 
   {
      while (argc > args)
      {
	 f_in.open(argv[args]);
	 {
	    cout << "Loading saved rule\n";
	    if (f_in)
	    {
	       for (int i = 0; i < RULELENGTH; ++i)
	       {
                  cell = (int) (f_in.get() - 48); 
		  pop[args-1].r[i] = cell;
		  pop[args].r[i] = cell;
	     //     cout << pop[0].r[i] << "   ";
	       }
	    }
	 }
	 f_in.close();
	 ++args;
      }
      for (int i = args; i < POPULATION; ++i) 
	 for (int j = 0; j < RULELENGTH; ++j)
	    pop[i].r[j] = rand() % STATES;
   }
   else*/    
      makeInitPop(); 
               
   evolAlg(0);
//   cout << endl << pop[POPULATION-1].r[RULELENGTH-1];

   return 0;
}


void makeInitPop()
{
   for (int i = 0; i < POPULATION; ++i) 
      for (int j = 0; j < RULELENGTH; ++j)
	 pop[i].r[j] = rand() % STATES;
}

void makeInitRand(int * initRand)
{
   //int background = rand() % STATES;
   //int foreground = rand() % STATES;

   // Random Init
   for (int i = 0; i < WIDTH; ++i)
      initRand[i] = rand() % STATES;

   // Single cell init
   //for (int i = 0; i < WIDTH; ++i)
   //   initRand[i] = background;
   //initRand[WIDTH/2]  = foreground;
}

void makeCAs(int * initRand)
{
   int ruleIndex[NEIGHBORHOOD];
   int start = - (NEIGHBORHOOD / 2);
   int finish = NEIGHBORHOOD / 2;
   int n = RULELENGTH - 1;

   for (int i = 0; i < POPULATION; ++i)
   {
      for (int j = 0; j < WIDTH; ++j)
	 ca[i].cell[j][0] = initRand[j];
      for (int j = 1; j < GENERATIONS; ++j)
	 for (int k = 0; k < WIDTH; ++k)
	 {
	    for (int l = start, count = 0; l <= finish; ++l, ++count)
	       ruleIndex[count] = ca[i].cell[(k + l) % WIDTH][j-1];
	    ca[i].cell[k][j] = pop[i].r[n - baseNtoDec(ruleIndex, STATES, NEIGHBORHOOD)];
         }
      //drawCA(ca[i], pop[i]);
   }
}

void drawCA(CA ca, rule r)
{
   cout << endl;
   for (int i = 0; i < RULELENGTH; ++i)
      cout << r.r[i];
   cout << endl << baseNtoDec(r.r, STATES, RULELENGTH);
   cout << endl << endl;
   for (int i = 0; i < GENERATIONS; ++i)
   {
      for (int j = 0; j < WIDTH; ++j)
         cout << ca.cell[j][i];
      cout << endl;
   }
   cout << endl;
}

long int baseNtoDec(int * num, int base, int length)
{
   int total = 0;
   int place = 1;
   int n = length - 1;

   for (int i = n; i >= 0; --i)
   { 
      total += place * num[i];
      place *= base;
   }
   return total;
}


 // Fitness I -- smallest largest area
void getFitness(fit * fitness)
{
   int size, largest[STATES];

   for (int i = 0; i < POPULATION; ++i)
   {
      for (int j = 0; j < WIDTH; ++j)
	 for (int k = 0; k < GENERATIONS; ++k)
	    visited[j][k] = false;
      for (int j = 0; j < STATES; ++j)
      {
	 largest[j] = 0;
	 for (int  k = 0; k < WIDTH; ++k)
	 {
	    for (int l = 0; l < GENERATIONS; ++l)
	    {
	       if (!visited[k][l]) 
	       {
		  size = getArea(k, l, j, i);
		  if (size > largest[j])
		     largest[j] = size;
	       }
	    }
         }
      }
      fitness[i].f = largest[0];
      fitness[i].ca = i;
      for (int j = 1; j < STATES; ++j)
         if (fitness[i].f > largest[j])
         //if (fitness[i].f < largest[j]) // edit: biggest
            fitness[i].f = largest[j];
   }
}

/*
// Fitness II -- largest average size regions
void getFitness(fit * fitness)
{
    int size;
    double largest[STATES]; // average
    //int num_area;
    //unsigned int total_size;
    double num_area, total_size, answer, large, small;
    int color_present[STATES];
    bool found;

#pragma omp parallel 
    {
#pragma omp for nowait    
    for (int i = 0; i < POPULATION; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
            for (int k = 0; k < GENERATIONS; ++k)
                visited[j][k] = false;
        for (int j = 0; j < STATES; ++j)
        {
            total_size = 0;
            num_area = 0;
            largest[j] = 0;
            large = 0;
            small = 1000;
            for (int l = 0; l < STATES; ++l)
                color_present[l] = 0;
            //	 for (int  k = 0; k < WIDTH; ++k)
            //	 {
            for (int l = 0; l < GENERATIONS; ++l) 
            {
                for (int  k = 0; k < WIDTH; ++k)
                {
                    if (!visited[k][l]) 
                    {
                        size = getArea(k, l, j, i);
                        if (size < 50) 
                            total_size += size;
                        //     cout << "Size: " << size << endl;
                        //		  if (size < 200)
                        if (size > 0)
                            ++num_area;

                        //if (size > largest[j])
                        //   largest[j] = size;
                    }
                    color_present[ca[i].cell[k][l]] = 1;

                }
                found = true;
                for (int k = 0; k < STATES; ++k)
                    if (color_present[k] == 0)
                        found = false;
                if (found == false)
                    total_size = 0;
            }
            //     cout<<"Total size:" << total_size << "    Num area:" << num_area<<endl;
            largest[j] =  (total_size * 2) / num_area;// + (total_size/100);
            if (largest[j] > large)
                large = largest[j];
            if (largest[j]  < small)
                small = largest[j];
            //largest[j] = (total_size/100);
        }
        //fitness[i].f = largest[0];
        answer = largest[0];
        fitness[i].ca = i;
        for (int j = 1; j < STATES; ++j)
            // if (fitness[i].f > largest[j]) 
            //	    fitness[i].f = largest[j];
            //fitness[i].f *= largest[j];
            answer *= largest[j] * 2 * (small / (large * large));
        //answer *= small / (10 *large);    
        fitness[i].f = (int)answer;
        }
    }
}
*/
/*
// Fitness III -- Smallest average area  
void getFitness(fit * fitness)
{
   int size, largest[STATES]; // average
   int smallest[STATES];
   int num_area;
   unsigned int total_size;

   for (int i = 0; i < POPULATION; ++i)
   {
      for (int j = 0; j < WIDTH; ++j)
	 for (int k = 0; k < GENERATIONS; ++k)
	    visited[j][k] = false;
      for (int j = 0; j < STATES; ++j)
      {
	 total_size = 0;
	 num_area = 0;
	 largest[j] = 0;
   smallest[j] = 1000;
	 for (int  k = 0; k < WIDTH; ++k)
	 {
	    for (int l = 0; l < GENERATIONS; ++l)
	    {
	       if (!visited[k][l]) 
	       {
		  size = getArea(k, l, j, i);
//		  if (size < 200)
//		     total_size += size;
//		  if (size < 200)
	//	     ++num_area;
       if (size < smallest[j])
          smallest[j] = size;
		  if (size > largest[j])
		     largest[j] = size;
	       }
	    }
         }
	 //largest[j] = (total_size*1000) / num_area;// + (total_size/100);
	 //largest[j] = (total_size/100);
      }
//      fitness[i].f = largest[0];
      fitness[i].f = 1000 - (largest[0] - smallest[0]);
 //     int big = 0;
  //    int little = largest[0];
      fitness[i].ca = i;
      for (int j = 1; j < STATES; ++j)
      {
//	 if (big < largest[i])
//	    big = largest[i];
//	 if (little > largest[i])
//	    little = largest[i];
//	 if (fitness[i].f < largest[j])

//	 if (fitness[i].f > largest[j])
//	    fitness[i].f = largest[j];
        fitness[i].f += 1000 - (largest[j] - smallest[j]);

      }
//      fitness[i].f /= STATES;
 //     float temp = (float)fitness[i].f;
  //    temp *= (float)little / (float)big;
   //   fitness[i].f = (int)temp;
   }
}
*/


int getArea(int x, int y, int state, int index)
{
   if (y < 0 || y == GENERATIONS)
      return 0;
   if (x < 0)
      return getArea(WIDTH-1, y, state, index);
   if (x == WIDTH)
      return getArea(0, y, state, index);
   if (ca[index].cell[x][y] != state || visited[x][y])
      return 0;
   visited[x][y] = true;
   return 1 + getArea(x+1, y, state, index) + getArea(x-1, y, state, index)
      + getArea(x, y+1, state, index) + getArea(x, y-1, state, index);
}


float displayFitness(fit * fitness)
{
   int total = fitness[0].f;
   int n = POPULATION / 2;
   float average;

   cout << endl << "[" << fitness[0].f;
   for (int i = 1; i < POPULATION; ++i)
   {
      cout << ", " << fitness[i].f;
      if (i < n )
	 total += fitness[i].f;
   }
   average = (float) total / (float) n;
   cout << "]\tAverage of fittest half: " << average << "\n";
   return average;
}

// Performs quick sort on a fitness array 
void sort(fit * fitness, int top, int bottom)
{
   int middle;
   if (top < bottom)
   {
      middle = partition(fitness, top, bottom);
      sort(fitness, top, middle);
      sort(fitness, middle + 1, bottom);
   }
}

// Quick sort partition
int partition(fit * fitness, int top, int bottom)
{
   fit x = fitness[top];
   int i = top -1;
   int j = bottom + 1;
   fit temp;
   do
   {
      do
	 --j;
      while (x.f > fitness[j].f);
      
      do
	 ++i;
      while (x.f < fitness[i].f);

      if (i < j)
      {
	 temp = fitness[i];
	 fitness[i] = fitness[j];
	 fitness[j] = temp;
      }
   }
   while (i < j);
   return j;
}

void mate(fit * fitness)
{
   int n = POPULATION / 4;
   int m = POPULATION / 2;
   int mom, dad, daughter, son, crossover;

   for (int i = 0; i < n; ++i) 
   {
      mom = fitness[i].ca;
      dad = fitness[2 * n - (i + 1)].ca;
      daughter = fitness[i + m].ca;
      son = fitness[i + m + n].ca;
      do
         crossover = rand() % RULELENGTH;
      while (crossover != 0 && crossover != RULELENGTH - 1);
      for (int j = 0; j < crossover; ++j)
      {
         pop[daughter].r[j] = pop[mom].r[j];
	 pop[son].r[j] = pop[dad].r[j];
      }
      for (int j = crossover; j < RULELENGTH; ++j)
      {
	 pop[daughter].r[j] = pop[dad].r[j];
	 pop[son].r[j] = pop[mom].r[j];
      }
   }
}


void mutate(fit * fitness)
{
   int changes = RULELENGTH / 1000 + 1;
   int n = POPULATION / 2;
   int m;

   for (int i = n; i < POPULATION; ++i)
   {
      m = fitness[i].ca;
      for (int j = 0; j < changes; ++j)
         pop[m].r[rand()  % RULELENGTH] = rand() % STATES; 
   }
}


void save(fit * fitness, int * bestFit, float * average)
{
   ofstream out; 
   int best = fitness[0].ca;
   //int best = fitness[1].ca;

   /*out.open("ca.txt");
   for (int i = 0; i < GENERATIONS; ++i)
      for (int j = 0; j < WIDTH; ++j)
	 out << ca[best].cell[j][i];
   out.close();
   out.open("best.txt");
   for (int i = 0; i < EPOCHS; ++i)
      out << bestFit[i] << endl;
   out.close();
   out.open("average.txt");
   for (int i = 0; i < EPOCHS; ++i)
      out << average[i] << endl;
   out.close();*/
   out.open(file_name);
   for (int i = 0; i < RULELENGTH; ++i)
      out << pop[best].r[i];
   out.close();
}


void evolAlg(int generation)
{
   int initRand[WIDTH];
   fit fitness[POPULATION];
   int best[EPOCHS];
   float average[EPOCHS];
   //int goal = 6000; //8750 / 4 STATES;
   int goal = 400; // 5 STATES;
   //int goal = 50; // neighborhood 5
   //int goal = 200;

//   makeInitRand(initRand);
   do
   {
      cout << "Generation: " << generation;
      makeInitRand(initRand);
      makeCAs(initRand);
      getFitness(fitness);
      sort(fitness, 0, POPULATION);
      average[generation] = displayFitness(fitness);
      best[generation] = fitness[0].f;
      mate(fitness);
      mutate(fitness);
      srand(time(NULL));
      ++generation;
   }
   while (generation < EPOCHS && fitness[0].f < goal);
   save(fitness, best, average);
}

