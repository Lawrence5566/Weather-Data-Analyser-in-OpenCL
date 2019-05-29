
void atomic_add_f(volatile global float* values, const float B)
//ints are faster, but we need floats for real accuracy
{
	union { float f; uint i; } oldVal; //share memory location to reduce memory use and not get
	union { float f; uint i; } newVal; //out of resources error
	do
	{
		oldVal.f = *values;
		newVal.f = oldVal.f + B;
	} while (atom_cmpxchg((volatile global uint*)values, oldVal.i, newVal.i) != oldVal.i);
}

void atomic_max_f(volatile global float* A, const float B)
{
	union { float f; uint i; } oldVal;
	union { float f; uint i; } newVal;
	do
	{
		oldVal.f = *A;
		newVal.f = max(oldVal.f, B);
	} while (atom_cmpxchg((volatile global uint*)A, oldVal.i, newVal.i) != oldVal.i);
}

void atomic_min_f(volatile global float* A, const float B)
{
	union { float f; uint i; } oldVal;
	union { float f; uint i; } newVal;
	do
	{
		oldVal.f = *A;
		newVal.f = min(oldVal.f, B);
	} while (atom_cmpxchg((volatile global uint*)A, oldVal.i, newVal.i) != oldVal.i);
}

kernel void reduce_add(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = N / 2; i > 0; i /= 2) { //using coalesced access , i = stride
		if (lid < i)					 //ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//copy the cache to output array
	if (!lid) {
		atomic_add_f(&B[0], scratch[lid]);
	}
}

/* //original reduce without coalessed access - can use this for demonstration of without coalessed access?
kernel void reduce_add(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}*/

kernel void reduce_min(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = N / 2; i > 0; i /= 2) { //using coalesced access , i = stride
		if (lid < i) {
			//check which is smaller
			if (scratch[lid + i] < scratch[lid])
				scratch[lid] = scratch[lid + i];
		}


		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//copy the cache to output array
	if (!lid) {
		atomic_min_f(&B[0], scratch[lid]);
	}
}

kernel void reduce_max(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = N / 2; i > 0; i /= 2) { //using coalesced access , i = stride
		if (lid < i) {
			if (scratch[lid + i] > scratch[lid])
				scratch[lid] = scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_max_f(&B[0], scratch[lid]);
	}
}

kernel void mean_variance_squared(global const float* A, global float* B, local float* scratch, float mean) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// calculate variance (input - mean))
	float variance = (A[id] - mean);

	//mean^2 + add to scratch 
	scratch[lid] = (variance * variance);

	barrier(CLK_LOCAL_MEM_FENCE); //sync after local memory copy

	for (int i = N/2; i > 0; i /= 2){
		if (lid < i)
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//if this is first element in local memory
	//atomic_add to put elements to output buffer
	if (!lid)
		atomic_add_f(&B[0], scratch[lid]);
}


//bitonic sort
//takes 116779 loops to sort

kernel void bitonic_sort_f(global const float* in, global float* out, local float* scratch, int merge)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int N = get_local_size(0);

	int max_group = (get_global_size(0) / N) - 1;
	int offset_id = id + ((N / 2) * merge);

	if (merge && gid == 0)
	{
		out[id] = in[id];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	scratch[lid] = in[offset_id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int l = 1; l < N; l <<= 1)
	{
		bool direction = ((lid & (l << 1)) != 0);

		for (int inc = l; inc > 0; inc >>= 1)
		{
			int j = lid ^ inc;
			float i_data = scratch[lid];
			float j_data = scratch[j];

			bool smaller = (j_data < i_data) || (j_data == i_data && j < lid);
			bool swap = smaller ^ (j < lid) ^ direction;

			barrier(CLK_LOCAL_MEM_FENCE);

			scratch[lid] = (swap) ? j_data : i_data;

			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	out[offset_id] = scratch[lid];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (merge && gid == max_group)
		out[offset_id] = in[offset_id];
}

///// int versions, gives exact results when multiply each input element by 10

kernel void bitonic_sort(global const int* in, global int* out, local int* scratch, int merge) {
	//takes 116779 loops to sort
	int id = get_global_id(0);
	int lid = get_local_id(0);

	//get the work group id
	int gid = get_group_id(0);

	int N = get_local_size(0);

	int maxGroup = (get_global_size(0) / N) - 1;

	//representation of the offset id based on the value of merge
	//when N = 1024, offset_id alternates between 0 and 512. 
	int offset_id = id + ((N / 2) * merge);

	// If merge and this is first group
	if (merge && gid == 0)
		out[id] = in[id];
	
	scratch[lid] = in[offset_id];

	//access local memory in a commutative manner, bitshifting i after each iteration for coalessed access instead of strides
	for (int l = 1; l < N; l <<= 1){
		//set the direction bool for this run of bitonic sort
		bool direction = ((lid & (l << 1)) != 0);

		for (int inc = l; inc > 0; inc >>= 1)
		{
			//gather the two data points to compare and store in i_data and j_data.
			int j = lid ^ inc;
			int i_data = scratch[lid];
			int j_data = scratch[j];

			//check if i_data < j_data and perform bitwise operations on the result combined with the direction, as well as 
			//whether j is within the work group, to determine if a swap should take place. 
			bool smaller = (j_data < i_data) || (j_data == i_data && j < lid);
			bool swap = smaller ^ (j < lid) ^ direction;

			//place the smallest value within the scratch buffer if swapping
			scratch[lid] = (swap) ? j_data : i_data;
		}
	}

	out[offset_id] = scratch[lid];

	//if a merge is taking place between groups and this group is the last, copy the last N/2 values from the input to the output buffer
	if (merge && gid == maxGroup)
		out[offset_id] = in[offset_id];
}

kernel void reduce_add_int(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = N / 2; i > 0; i /= 2) { //using coalesced access , i = stride
		if (lid < i)					 //ensure that i does not decrement beyond the local id when performing addition of current and next scratch element.
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

kernel void reduce_min_int(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = N / 2; i > 0; i /= 2) { //using coalesced access , i = stride
		if (lid < i) {
			//check which is smaller
			if (scratch[lid + i] < scratch[lid])
				scratch[lid] = scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//copy the cache to output array
	if (!lid) {
		atomic_min(&B[0], scratch[lid]);
	}
}

kernel void reduce_max_int(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = N / 2; i > 0; i /= 2) { //using coalesced access , i = stride
		if (lid < i) {
			if (scratch[lid + i] > scratch[lid])
				scratch[lid] = scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_max(&B[0], scratch[lid]);
	}
}

kernel void mean_variance_squared_int(global const int* A, global int* B, local int* scratch, float mean) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// calculate variance (input - mean))
	float variance = (A[id] - mean);

	//mean^2 + add to scratch 
	scratch[lid] = (int)(variance * variance) / 10; //10 as mean is 10* bigger

	barrier(CLK_LOCAL_MEM_FENCE); //sync after local memory copy

	for (int i = N / 2; i > 0; i /= 2) {
		if (lid < i)
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//if this is first element in local memory
	//atomic_add to put elements to output buffer
	if (!lid)
		atomic_add_f(&B[0], scratch[lid]);
}
