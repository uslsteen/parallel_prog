#include "merge_sort.h"

int main(int argc, char** argv) {

	int size = atoi(argv[1]);

	int *arr = (int*)calloc(size, sizeof(int));
	assert(arr);
	input(argv[2], arr, size);

#if 0	
	make_random_arr(arr, size);
#endif

	int *lin_arr = (int*)calloc(size, sizeof(int));
	memcpy(lin_arr, arr, size);
	do_linear(lin_arr, size);

	int *par_arr = (int*)calloc(size, sizeof(int));
	memcpy(par_arr, arr, size);

	do_parallel(argc, argv, arr, size);	

	output(arr, size, "sorted array:\n");

	free(arr);
	free(par_arr);
	free(lin_arr);

	return 0;
}