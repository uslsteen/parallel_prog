#ifndef MERGE_SORT_H
#define MERGE_SORT_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

void merge_sort(int* arr, int left, int right);
void merge(int* arr, int left, int middle, int right);

void do_parallel(int argc, char** argv, int* arr, int orig_size);
void do_linear(int *arr, int orig_size);

int output(int* arr, int len, char* descr);
int input(char* filename, int* arr, int len);

void make_random_arr(int* arr, int len);

#endif // MERGE_SORT_H