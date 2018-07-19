#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono>


#define size_data 32
#define upper_bit 6
#define lower_bit 0




__global__ void Radix(int* keys, int* values) {



	__shared__ volatile int keys_1[size_data * 2];
	__shared__ volatile int values_1[size_data * 2];



	keys_1[threadIdx.x] = keys[threadIdx.x];
	values_1[threadIdx.x] = values[threadIdx.x];

	unsigned int bit_mask = 1 << lower_bit;
	unsigned int offset = 0;
	unsigned int mask = 0xFFFFFFFFU << threadIdx.x;
	unsigned int pos_index;

	for (int i = lower_bit; i <= upper_bit; i++) {



		int temp_keys = keys_1[((size_data - 1) - threadIdx.x) + offset];
		int temp_values = values_1[((size_data - 1) - threadIdx.x) + offset];
		unsigned int current_bit = temp_keys&bit_mask;


		unsigned int ones = __ballot(current_bit);
		unsigned int zeroes = ~ones;


		offset ^= size_data;

		if (!current_bit)
		{
			pos_index = __popc(zeroes&mask);
		}
		else {
			pos_index = __popc(zeroes) + __popc(ones&mask);
		}

		keys_1[pos_index - 1 + offset] = temp_keys;
		values_1[pos_index - 1 + offset] = temp_values;

		bit_mask <<= 1;
	}


	// copy back the result
	keys[threadIdx.x] = keys_1[threadIdx.x + offset];
	// copy back the result
	values[threadIdx.x] = values_1[threadIdx.x + offset];


}


__global__ void rle(int* d_data, int* d_run, int* d_Cell_id) {

	int i = threadIdx.x;
	unsigned int laneid = i & 0x1f;
	int val = d_data[i];
	int nval = __shfl_down(val, 1);
	unsigned int mask = __ballot(nval != val);
	int offset = __popc(mask & ((1 << laneid) - 1));
	int zcnt = __clz(mask & ((1 << laneid) - 1));
	int runcnt = zcnt - 31 + laneid;


	if (nval != val || i == 31) {

		d_Cell_id[offset] = val;
		d_run[offset] = runcnt;

	}


}

int main() {



	
	int NUMBER_OF_PARTICLES = 32;
	int NUMBER_OF_KEYS = 16;
	int keys[32];
	int value[32];
	int *d_keys;
	int *d_values;
	int *d_Count;
	int *d_Cell_id;
	int Count[16];
	int Cell_id[16];
	int offset = 0;
	int counter = 0;

	srand(time(NULL));
	for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
		keys[i] = rand() % NUMBER_OF_KEYS;
		value[i] = i;
	}


	//allocate memory for keys
	cudaMalloc((void**)&d_keys, NUMBER_OF_PARTICLES * sizeof(int));
	//copy keys to shares memory
	cudaMemcpy(d_keys, keys, NUMBER_OF_PARTICLES * sizeof(int), cudaMemcpyHostToDevice);
	//allocate memory for values
	cudaMalloc((void**)&d_values, NUMBER_OF_PARTICLES * sizeof(int));
	//copy values to shared memory
	cudaMemcpy(d_values, value, NUMBER_OF_PARTICLES * sizeof(int), cudaMemcpyHostToDevice);

	Radix << <1, NUMBER_OF_PARTICLES >> > (d_keys, d_values);
	cudaMalloc((void**)&d_Count, NUMBER_OF_KEYS * sizeof(int));
	cudaMalloc((void**)&d_Cell_id, NUMBER_OF_KEYS * sizeof(int));

	rle << <1, NUMBER_OF_PARTICLES >> > (d_keys, d_Count, d_Cell_id);

	cudaMemcpy(value, d_values, NUMBER_OF_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Cell_id, d_Cell_id, NUMBER_OF_KEYS * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Count, d_Count, NUMBER_OF_KEYS * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(keys, d_keys, NUMBER_OF_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_keys);
	cudaFree(d_values);
	cudaFree(d_Count);

	for (int i = 0; i < NUMBER_OF_KEYS; i++) {
		if (i != Cell_id[counter]) {// empty cell

			printf("Cell id: %2d  Count: %d\n", i, 0);
		}
		else {

			printf("Cell id: %2d  Count: %d	Particle id: ", Cell_id[counter], Count[counter]);

			for (int k = 0; k < Count[counter]; k++) {

				printf(" %d", value[offset]);
				offset += 1;
			}
			printf("\n");
			counter += 1;

		}








	}

	return 0;
}
