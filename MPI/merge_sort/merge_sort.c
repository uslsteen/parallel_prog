#include "merge_sort.h"

void do_parallel(int argc, char** argv, int* arr, int orig_size) {
	int rank = 0, commsize = 0, root = 0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	
	//! divide the array in equal-sized chunks
	int chunk_size = orig_size / commsize;
	
	//! send each subarray to each process
	int *sub_array = (int *)calloc(chunk_size, sizeof(int));
	assert(sub_array);

	MPI_Scatter(arr, chunk_size, MPI_INT, sub_array, chunk_size, MPI_INT, root, MPI_COMM_WORLD);
	
	//! perform the merge_sort on each process
	merge_sort(sub_array, 0, (chunk_size - 1));
	
	//! gather the sorted subarrays into one
	int *sorted = NULL;

	if(rank == 0)
		sorted = (int*)calloc(orig_size, sizeof(int));		

	MPI_Gather(sub_array, chunk_size, MPI_INT, sorted, chunk_size, MPI_INT, root, MPI_COMM_WORLD);
	
	//! do the final merge_sort call
	if (rank == 0) {
	    double start = MPI_Wtime();

		merge_sort(sorted, 0, (orig_size - 1));

		double end = MPI_Wtime();
		printf("Executed time of parallel %lf seconds\n", end - start);
		free(sorted);
	}

    
	free(sub_array);
	MPI_Finalize();
}

void do_linear(int *arr, int orig_size) {
    clock_t start = clock();
	merge_sort(arr, 0, orig_size - 1);
    clock_t end = clock();

	printf("Executed time  of linear %lf seconds\n", (double)(end - start)/CLOCKS_PER_SEC);
}

void merge(int* arr, int left, int middle, int right)
{
	int i = 0, j = 0, k = 0;
	int size_l = middle - left + 1;
	int size_r = right - middle;

	/* create temp buffers */

	int* temp_L = NULL, * temp_R = NULL;

	temp_L = (int*)calloc(size_l, sizeof(temp_L[0]));
	assert(temp_L);

	temp_R = (int*)calloc(size_r, sizeof(temp_R[0]));
	assert(temp_R);

	/* Copy data to temp buffers temp_L[size_l] and temp_R[size_r] */
	for (i = 0; i < size_l; i++)
		temp_L[i] = arr[left + i];

	for (j = 0; j < size_r; j++)
		temp_R[j] = arr[middle + 1 + j];

	/* Merge the temp arrays back into arr[l..r]*/

	i = 0; // Initial index of first temp_buf
	j = 0; // Initial index of second temp_buf
	k = left; // Initial index of merged buffer
	while (i < size_l && j < size_r)
	{
		if (temp_L[i] <= temp_R[j]) {
			arr[k] = temp_L[i];
			i++;
		}
		else {
			arr[k] = temp_R[j];
			j++;
		}
		k++;
	}

	/* Copy the remaining elements of temp_L[], if there are any */
	while (i < size_l) {
		arr[k] = temp_L[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of temp_R[], if there are any */
	while (j < size_r) {
		arr[k] = temp_R[j];
		j++;
		k++;
	}

	free(temp_L);
	free(temp_R);
}

void merge_sort(int* arr, int left, int right)
{
	int middle = (left + right)/2;

	if (left >= right)
		return;

	merge_sort(arr, left, middle);
	merge_sort(arr, middle + 1, right);

	merge(arr, left, middle, right);
}

int input(char* filename, int* arr, int len) {
	assert(arr);

	FILE *file = NULL;
     
    if ((file= fopen(filename, "r")) == NULL) {
        perror("Error occured while opening file");
        return -1;
    }

	for (int i = 0; i < len; ++i) {
		int check = fscanf(file, "%d", (arr + i));
		assert(check);

		// printf("%d ", arr[i]);
	}

	fclose(file);
}

int output(int* arr, int len, char* descr) {
	assert(arr);
	assert(descr);

    FILE *file = NULL;
     
    if ((file= fopen("sorted_arr.txt", "w")) == NULL) {
        perror("Error occured while opening file");
        return -1;
    }

    fprintf(file, "%s", descr);
    for(int id = 0; id < len; ++id)
        fprintf(file, "%d ", arr[id]);

	fclose(file);
}


void make_random_arr(int* arr, int len) {
	assert(arr);

	srand(time(NULL));
	
	for(uint32_t id = 0; id < len; ++id)	
		arr[id] = rand() % len;
}