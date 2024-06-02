﻿// FastImagConvolutioin.h : включаемый файл для стандартных системных включаемых файлов
// или включаемые файлы для конкретного проекта.

#pragma once

#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>

#define KERNX 7     //MUST BE ODD
#define KERNY 7     //MUST BE ODD

// If you want to modify the kernel, you can do it here! Don't worry about normalizing it, it will be done for you automatically.

//Comment this section out if you don't want to use random kernels
/*
extern const int kernel_x = KERNX, kernel_y = KERNY;
extern float kernel[KERNX*KERNY];
*/

// kernel for edge detection
int kernel_x = 3, kernel_y = 3;

float kernel[] = { -1, -1, -1,
                   -1, 8, -1,
                   -1, -1, -1 };
//int kernel_x = 3, kernel_y = 3;

int minimum = 400, maximum = 1201, step = 200;

void generate_kernel();

void normalize(float* kernel);


int conv2D(float* in, float* out, const int data_size_X, const int data_size_Y,
    float* kernel, int kernel_x, int kernel_y);

int referenceConv2D(float* in, float* out, int data_size_X, int data_size_Y,
    float* kernel, int kernel_x, int kernel_y);


// TODO: установите здесь ссылки на дополнительные заголовки, требующиеся для программы.
