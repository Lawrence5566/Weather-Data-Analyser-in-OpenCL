#pragma comment(lib, "OpenCL.lib")

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"
#include "DataAnalyser.h"

typedef float mytype;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void ReadTempValues(vector<mytype> *tempArray) {
	string line;
	ifstream file("Textfiles/temp_lincolnshire.txt");
	if (file.is_open()) {
		while (getline(file, line)) {
			istringstream iss(line); //use istringstream on line to split data
			vector<string> tokens{ istream_iterator<string>{iss},
							istream_iterator<string>{} };

			float item = stof(tokens[tokens.size() - 1]); //convert string to float
			//int item = (int)stof(tokens[tokens.size() - 1])*10 //or convert them to ints
			tempArray->push_back(item);
		}
		file.close();
	}

	cout << "data read" << endl;

}

cl::Context context;
cl::CommandQueue queue;
cl::Program::Sources sources; //load & build the device code
cl::Program program;

size_t local_size = 32; //32 work group size (prefered kernel size)

float mean = 0.f;

void MinMaxAvg(vector<mytype> arr) {
	std::vector<mytype> B(1); //B is used as single answer buffer
	std::vector<mytype> C(1);
	std::vector<mytype> D(1);

	size_t input_elements = arr.size();//number of input elements
	size_t input_size = arr.size() * sizeof(mytype);//size in bytes
	size_t output_size = B.size() * sizeof(mytype);	//size in bytes

	//device - buffers
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

	//create profiling event for run time
	cl::Event prof_event_avg;
	cl::Event prof_event_min;
	cl::Event prof_event_max;
	cl::Event write_event;

	//copy array A to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &arr[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

	//setup and execute all kernels (i.e. device code)
	cl::Kernel kernel_avg = cl::Kernel(program, "reduce_add");
	kernel_avg.setArg(0, buffer_A);
	kernel_avg.setArg(1, buffer_B);
	kernel_avg.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size for kernel#

	queue.enqueueNDRangeKernel(kernel_avg, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_avg); //extra profiling arguments
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

	cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
	kernel_min.setArg(0, buffer_A);
	kernel_min.setArg(1, buffer_B);
	kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size for kernel

	queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_min);
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &C[0]);

	cl::Kernel kernel_max = cl::Kernel(program, "reduce_max");
	kernel_max.setArg(0, buffer_A);
	kernel_max.setArg(1, buffer_B);
	kernel_max.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size for kernel

	queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_max);
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &D[0]);

	cl_ulong writetime = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	//calculate results

	mean = B[0] / arr.size();		//divide by 10 if using ints
	float min = C[0];
	float max = D[0];

	std::cout << "average mean value = " << mean << std::endl;
	std::cout << "min value = " << min << std::endl;
	std::cout << "max value = " << max << std::endl;

	std::cout << "write time [ns]" << writetime << endl;

	std::cout << "kernel_avg execution time [ns]:" << prof_event_avg.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_avg.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	std::cout << "kernel_min execution time [ns]:" << prof_event_min.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_min.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	std::cout << "kernel_max execution time [ns]:" << prof_event_max.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_max.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

	cl::flush();
}

void Std(vector<mytype> arr) {
	size_t input_elements = arr.size();//number of input elements
	size_t input_size = arr.size() * sizeof(mytype);//size in bytes

	//create output vectors

	std::vector<mytype> std(1);
	std::vector<mytype> std_mean_vars(input_elements);

	size_t output_size = std.size() * sizeof(mytype);	//size in bytes

	//device - buffers
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_std_mean_vars(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer buffer_std(context, CL_MEM_READ_WRITE, output_size);

	//create profiling events for run time
	cl::Event prof_event_mean;
	cl::Event prof_event_add;
	cl::Event write_event;

	//copy array A to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &arr[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_std_mean_vars, 0, 0, input_size);
	queue.enqueueFillBuffer(buffer_std, 0, 0, output_size);

	//work out standard dev
	cl::Kernel kernel_mean_var = cl::Kernel(program, "mean_variance_squared");
	kernel_mean_var.setArg(0, buffer_A);
	kernel_mean_var.setArg(1, buffer_std_mean_vars);
	kernel_mean_var.setArg(2, cl::Local(local_size * sizeof(mytype))); //local memory size for kernel
	kernel_mean_var.setArg(3, mean);

	cl::Kernel kernel_std_add = cl::Kernel(program, "reduce_add");
	kernel_std_add.setArg(0, buffer_std_mean_vars);
	kernel_std_add.setArg(1, buffer_std);
	kernel_std_add.setArg(2, cl::Local(local_size * sizeof(mytype)));

	queue.enqueueNDRangeKernel(kernel_mean_var, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_mean);
	queue.enqueueNDRangeKernel(kernel_std_add, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_add);

	queue.enqueueReadBuffer(buffer_std_mean_vars, CL_TRUE, 0, input_size, &std_mean_vars[0]);
	queue.enqueueReadBuffer(buffer_std, CL_TRUE, 0, output_size, &std[0]);

	cl_ulong writetime = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	std::cout << std::endl;
	std::cout << "standard dev = " << sqrt(std[0] / input_elements) << std::endl;

	//display kernel execution time and write times
	std::cout << "write time [ns]:" << writetime << std::endl;
	std::cout << "kernel_mean_var execution time [ns]:" << (prof_event_mean.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_mean.getProfilingInfo<CL_PROFILING_COMMAND_START>()) +
		(prof_event_add.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_add.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

	cl::flush();
}

bool Sorted(vector<mytype> arr)
{
	//sequentially check whether or not an array is sorted
	for (int i = 0; i < arr.size() - 1; i++){
		if (arr[i] > arr[i + 1])
			return false;
	}

	return true;
}

void Sort(vector<mytype> arr) { //sorting, median and quartiles
	cout << " \nSorting" << endl;

	size_t input_elements = arr.size();//number of input elements
	size_t input_size = arr.size() * sizeof(mytype);//size in bytes

	std::vector<mytype> sortedList = arr; //set to be same as array for start

	cl::Buffer buffer_sorted(context, CL_MEM_READ_WRITE, input_size); //starts with array input
	cl::Buffer buffer_sorted2(context, CL_MEM_READ_WRITE, input_size);

	cl::Event write_event;
	cl::Event prof_event_sort;

	queue.enqueueWriteBuffer(buffer_sorted, CL_TRUE, 0, input_size, &arr[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_sorted2, 0, 0, input_size);

	cl_ulong runtime = 0;
	
	//bitonic sort//

	cl::Kernel kernel_sort = cl::Kernel(program, "bitonic_sort_f");
	kernel_sort.setArg(0, buffer_sorted);
	kernel_sort.setArg(1, buffer_sorted2);
	kernel_sort.setArg(2, cl::Local(local_size * sizeof(mytype)));
	kernel_sort.setArg(3, 0);

	queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_sort);

	int i = 0;
	while (!Sorted(sortedList))
	{
		queue.enqueueWriteBuffer(buffer_sorted, CL_TRUE, 0, input_size, &sortedList[0]);
		kernel_sort.setArg(3, (i++ % 2));

		queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_sort);
		queue.enqueueReadBuffer(buffer_sorted2, CL_TRUE, 0, input_size, &sortedList[0]);

		runtime = runtime + prof_event_sort.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_sort.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		cl::flush(); //flush queue
	}

	queue.enqueueReadBuffer(buffer_sorted2, CL_TRUE, 0, input_size, &sortedList[0]);

	cl_ulong writetime = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	std::cout << "1QT: = " << sortedList[(sortedList.size() * (1.0f / 4.0f))] << std::endl;
	std::cout << "3QT: = " << sortedList[(sortedList.size() * (3.0f / 4.0f))] << std::endl;
	std::cout << "median: = " << sortedList[(sortedList.size() / 2) - 1] << std::endl;

	std::cout << "buffer write time [ns]:" << writetime << std::endl;
	std::cout << "kernel_sort execution time [ns]:" << runtime << std::endl;

	cl::flush();
}

 
int main(int argc, char **argv) {
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		// host operations
		//Select computing devices
		context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl << std::endl;

		//create a queue to which we will push commands for the device
		queue = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE);

		//load & build the device code

		AddSources(sources, "kernels.cl");

		program = cl::Program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//allocate and load txt file into memory
		vector<mytype>* tempArray = new vector<mytype>();
		ReadTempValues(tempArray);		//get data from text file

		std::vector<mytype> A = *tempArray;
		
		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<mytype> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		//call functions
		MinMaxAvg(A);
		Std(A);
		Sort(A);

		string x;
		std::cin >> x;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}

//get prefered worksgroup sizes - not needed? effects padding?

	/*
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; //get device
	size_t local_size_kernel_avg = kernel_max.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>
		(device);

	std::cout << local_size_kernel_avg << endl;
	*/

//size_t nr_groups = input_elements / local_size;
