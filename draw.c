
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <openssl/rand.h>

const int R 			= 8;  //8
const int I 			= 2;
const int DIFFUSE_LENGTH 	= 40; //40
const int INPUT_LENGTH 		= 4;
const int STATES 		= 2; 
const int NEIGHBORHOOD 		= 3;
const int RULELENGTH 		= pow(STATES, NEIGHBORHOOD);
//const int WIDTH			= DIFFUSE_LENGTH * R;
const int READOUT_LENGTH	= R * DIFFUSE_LENGTH * I;
const int DISTRACTOR_PERIOD	= 200;
// For 5-bit memory task
const int SEQUENCE_LENGTH	= DISTRACTOR_PERIOD + 10;
const int TEST_SETS		= 32;

const int WIDTH = 800; //SEQUENCE_LENGTH * TEST_SETS; //400
//const int GENERATIONS = WIDTH; //SEQUENCE_LENGTH * I * TEST_SETS; //READOUT_LENGTH; //200
const int GENERATIONS = 800; //READOUT_LENGTH; //200

int main(int argc, char* argv[])
{
   int i, j, k, l, states = 4; // 4;
   if (argc != 3)
   {
      printf("Usage: draw <CA file> <PPM file>\n");
      return 1;
   }

   FILE * f_out = fopen(argv[2],"w");

   fputs("P3\n", f_out);
   fprintf(f_out, "%d %d\n", 3 * WIDTH, 3 * GENERATIONS);
   //fputs("1200 600\n", f_out);
   fputs("255\n", f_out);
   
   FILE * f_in = fopen(argv[1], "r");
   int colorNum = 3 * states;
   unsigned char colors[3*states];
   unsigned char newColor[1];
   // Set all colors randomly
   RAND_bytes(colors, colorNum);

   // Set range of r value for each color (50i -> 50(i+i) - 1)
   for (i = 0; i < states; ++i)
   {
      while ((int) *(colors+(3*i)) <  50*i || (int) *(colors+(3*i)) >= 50*(i+1))
      {
	 RAND_bytes(newColor, 1);
	 *(colors+(3*i)) = *newColor;
      }
   }

   // Set equal numbers of least bit of each color
   for (i = 0; i < colorNum; ++i)
   {
      if (i % 2 == 0)
	 colors[i] = colors[i] & ~1;
      else 
	 colors[i] = colors[i] | 1;
   }
   
   int  state;
   char charState;
   int layer[WIDTH];
   for (i = 0; i < GENERATIONS; ++i)
   {
      for (j = 0; j < WIDTH; ++j)
      {
         fscanf(f_in, " %c", &charState);
         state = (int) charState - 48;
         //printf("%d ",state);
         layer[j] = state;
         for (k = 0; k < 3; ++k)
            fprintf(f_out, "%d %d %d ", *(colors+state*3),*(colors+state*3+1),*(colors+state*3+2));


         if (i % 3 == 2)
            fprintf(f_out, "\n");
      }
      for (j = 0; j < 2; ++j)
       for (l = 0; l < WIDTH; ++l)
          for (k = 0; k < 3; ++k)
             fprintf(f_out, "%d %d %d ", *(colors+layer[l]*3),*(colors+layer[l]*3+1),*(colors+layer[l]*3+2));
   }
   fclose(f_in);
   fclose(f_out);


   return 0;
}





