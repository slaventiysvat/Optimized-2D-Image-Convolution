// FastImagConvolutioin.cpp: определяет точку входа для приложения.
//



#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include "./ImageConvolution/inc/FastImagConvolutioin.hpp"

// MSVC defines this in winsock2.h!?
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((uint64_t)file_time.dwLowDateTime);
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec = (long)((time - EPOCH) / 10000000L);
    tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
    return 0;
}

// kernel for edge detection
//extern int kernel_x;
//extern int kernel_y;
//extern float kernel[];

//int kernel_x = 3, kernel_y = 3;

//extern int minimum;
//extern int maximum;
//extern int step;

static int SIZE_DIM;

int color_table_size, color_size, x_dim, y_dim;
unsigned char* color_table;

float* in_image, * out_image, * out_ref_image, * in_image1, * in_image2, * in_image3;
int real_image = 0, repeat = 0;
unsigned char info[54];

int main( int argc, char ** argv ) {
  // Normalize the kernel so that it produces correct colors.
  //generate_kernel();   //Comment this out if you don't want to use randomly generated kernels
  printf("Kernel:\n");
  for (int y = 0; y < kernel_y; y++){
 	for ( int x = 0; x < kernel_x; x++){
        	printf("%0.f\t", kernel[x + y * kernel_x]);
        }
	printf("\n");
  }
  printf("\n");

  normalize(kernel);

  //if ( argc == 3 ) {
  //  x_dim = atoi(argv[1]);
  //  y_dim = atoi(argv[2]);
  //  repeat = 1;
  //} else if ( argc == 2 ) {
    real_image = 1;
    repeat = 1;

    //FILE* file_in = fopen(argv[1], "rb");
    FILE* file_in = fopen("./Images/image0.bmp", "rb");
    int check = fread(info, sizeof(unsigned char), 54, file_in); // Read the 54-byte info header
    color_table_size = *((int*)(info + 0x0a))- 54;
    color_table = (unsigned char *) malloc(sizeof(unsigned char) * color_table_size);
    check = fread(color_table, sizeof(unsigned char), color_table_size, file_in);

    color_size = *((int*)(info + 0x1c));
    if (*((int*)(info + 0x0e)) != 40 || (color_size != 8 && color_size != 32) ) {
      printf( "Sorry this file format is not supported yet. Please use a different image!\nBMP pictures with 8 bits per pixel grayscale (may be different from seemingly grayscale pictures that actually uses 4 colors) and 32 bit per pixel color image.\n" );
      return -1;
    }
    color_size /= 8;

    x_dim = *((int*)(info + 0x12));
    y_dim = *((int*)(info + 0x16));
    x_dim = (x_dim > 0) ? x_dim : -1 * x_dim;
    y_dim = (y_dim > 0) ? y_dim : -1 * y_dim;

    SIZE_DIM = x_dim * y_dim;

    //unsigned char data[size*color_size];

    unsigned char * data = (unsigned char*) malloc(sizeof(unsigned char) * (SIZE_DIM * color_size));

    check = fread(data, sizeof(unsigned char), SIZE_DIM * color_size, file_in);
    fclose(file_in);
    in_image = (float *) malloc(sizeof(float) * SIZE_DIM);
    if (color_size > 1) {
      in_image1 = (float *) malloc(sizeof(float) * SIZE_DIM);
      in_image2 = (float *) malloc(sizeof(float) * SIZE_DIM);
      in_image3 = (float *) malloc(sizeof(float) * SIZE_DIM);
    }
    out_image = (float *) malloc(sizeof(float) * SIZE_DIM);
    out_ref_image = (float *) malloc(sizeof(float) * SIZE_DIM);

    for (int i = 0; i < SIZE_DIM; i++) {
      in_image[i] = ((float) data[i*color_size])/255;
      if (color_size > 1) {
        in_image1[i] = ((float) data[i*color_size+1]/255);
        in_image2[i] = ((float) data[i*color_size+2]/255);
        in_image3[i] = ((float) data[i*color_size+3]/255);
      }
    }
  //}

  double total_Gflop_s = 0;
  double total_iterations = 0;
  for ( int counter = 0; counter < 10 && (repeat || (!counter)); counter++ ) {
    for ( int x = minimum; x < maximum; x+=step ) {
      for ( int y = minimum; y < maximum; y+=step ) {
        if (repeat) {
          x = x_dim;
          y = y_dim;
          maximum = ( x_dim > y_dim ) ? y_dim-1 : x_dim-1;
        }
        //if (!real_image) {
        //  in_image = (float*) malloc( x * y * sizeof(float) );
        //  out_image = (float*) malloc( x * y * sizeof(float) );
        //  out_ref_image = (float*) malloc( x * y * sizeof(float) );
        //  for( int i = 0; i < x * y; i++ ) in_image[i] = 2 * drand48() - 1;
        //}

        // Clear the memory of reference and output to be used to check for correctness.
        memset( out_ref_image, 0, sizeof(float) * x * y );
        referenceConv2D( in_image, out_ref_image, x, y, kernel, kernel_x, kernel_y);
        memset( out_image, 0, sizeof(float) * x * y );
        conv2D( in_image, out_image, x, y, kernel, kernel_x, kernel_y ); 
        // Subtract out the difference to figure out how large it is. If the difference is large enough, then error out.
        for ( int i = 0; i < x * y; i++ ) {
          float temp = out_image[i] - out_ref_image[i];
//        if (counter ==0 && x == minimum && y == minimum)
//          printf("Ref: %f, Out: %f\n", out_ref_image[i], out_image[i]);
          if ( temp * temp > 0.0001 ) {
            printf( "FAILURE: error in convolution calculation exceeds an acceptable margin.\n" );
            printf( "Expected: %f, received: %f, with image size x = %d, y = %d and kernel size x = %d, y = %d at array location: %d\n", out_ref_image[i], out_image[i], x, y, kernel_x, kernel_y, i );
            return -1;
          }
        }

        double Gflop_s, seconds = -1.0;
        for( int n_iterations = 1; seconds < 0.1; n_iterations *= 2) {
          /* warm-up */
          conv2D( in_image, out_image, x, y, kernel, kernel_x, kernel_y); 
          /* measure time */
          struct timeval start, end;
          gettimeofday( &start, NULL );
          for (int i = 0; i < n_iterations; i++) {

              conv2D( in_image, out_image, x, y, kernel, kernel_x, kernel_y); 
          }
            
          gettimeofday( &end, NULL );
          seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
 
        /* compute Gflop/s rate */
          Gflop_s = 2e-9 * n_iterations * 2 * x * y * kernel_x * kernel_y / seconds;
        }
        printf( "Image Dimemsions: x = %d, y = %d \t Kernel Dimensions: x = %d, y = %d \t Performance: %g Gflop/s\n", x, y, kernel_x, kernel_y, Gflop_s );
        total_Gflop_s += Gflop_s;
        total_iterations++;

        if (!real_image) {
          free(in_image);
          free(out_image);
          free(out_ref_image);
        }
      } // End of y loop
    } // End of x loop
  } // End of counter loop
  double average = total_Gflop_s / total_iterations;
  printf( "Average Gflop/s = %g\n", average );

  if (real_image) {

   /* unsigned char data[SIZE_DIM *color_size];
    float out_image1[SIZE_DIM], out_image2[SIZE_DIM];*/

    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * (SIZE_DIM * color_size));
    float* out_image1 = (float*)malloc(sizeof(float) * (SIZE_DIM));
    float* out_image2 = (float*)malloc(sizeof(float) * (SIZE_DIM));

    memset( out_image, 0, sizeof(float) * SIZE_DIM);
    memset( out_image1, 0, sizeof(float) * SIZE_DIM);
    memset( out_image2, 0, sizeof(float) * SIZE_DIM);
    conv2D( in_image, out_image, x_dim, y_dim, kernel, kernel_x, kernel_y ); 
    if (color_size > 1) {
      conv2D( in_image1, out_image1, x_dim, y_dim, kernel, kernel_x, kernel_y); 
      conv2D( in_image2, out_image2, x_dim, y_dim, kernel, kernel_x, kernel_y); 
    }

    for (int i = 0; i < SIZE_DIM; i++) {
      data[i*color_size] = (unsigned char) (out_image[i] * 255);
      if (color_size > 1) {
        data[i*color_size+1] = (unsigned char) (out_image1[i] * 255);
        data[i*color_size+2] = (unsigned char) (out_image2[i] * 255);
        data[i*color_size+3] = (unsigned char) (in_image3[i] * 255);
      }
    }
    FILE* f = fopen("out_img.bmp", "wb");
    fwrite(info, sizeof(unsigned char), 54, f);
    fwrite(color_table, sizeof(unsigned char), color_table_size, f);
    fwrite(data, sizeof(unsigned char), SIZE_DIM*color_size, f);
    fclose(f);
    
    free(in_image);
    free(out_image);
    free(out_ref_image);
    free(color_table);
    if (color_size > 1) {
      free(in_image1);
      free(in_image2);
      free(in_image3);
    }
  } // End of real image storing
  return 0;
} // End of main