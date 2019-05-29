#pragma comment(lib, "OpenCL.lib")

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.hpp>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void ReadTempValues(vector<float> *tempArray) {
	string line;
	ifstream file("Textfiles/temp_lincolnshire_short.txt");
	if (file.is_open()) {
		while (getline(file, line)) {
			istringstream iss(line); //use istringstream on line to split data
			vector<string> tokens{ istream_iterator<string>{iss},
							istream_iterator<string>{} };

			float item = stof(tokens[tokens.size() - 1]); //convert string to float
			tempArray->push_back(item); //add last item (temp) to array
		}
		file.close();
	}

}

int main(int argc, char **argv) {

	typedef int mytype;

	//allocate and load txt file into memory
	//vector<mytype>* tempArray = new vector<mytype>(20, 1);

	//ReadTempValues(tempArray);//get data from text file

	// - handle command line options such as device selection, verbosity, etc. -
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//- host operations -
		// Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device & enable profiling
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

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

		// - memory allocation - //
		// (temp values done above already)
		std::vector<mytype> tempArray(20, 1);

		size_t input_elements = tempArray.size();//number of elements
		size_t input_size = tempArray.size() * sizeof(mytype);//size in bytes

		size_t local_size = 2; //custom work group size

		//padding stuff if we are using manual work group size
		size_t padding_size = tempArray.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			tempArray.insert(tempArray.end(), A_ext.begin(), A_ext.end());
		} 

		//host - output
		std::vector<float> B(1);
		size_t output_size = B.size() * sizeof(float);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size); 

		// - device operations - 

		//copy array tempArray into device memory and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &tempArray[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		// Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "reduce_add_3");
		kernel.setArg(0, buffer_A);
		kernel.setArg(1, buffer_B);
		kernel.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size for reduceAdd

		//create profiling event for run time
		cl::Event prof_event;

		/* //prefered work group size stuff
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; //get device
	cerr << kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>
		(device) << endl; //get prefered work group size info
	*/

		//4th arguements specifies workgroup size -  cl::NDRange(local_size) is manual
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size),
			NULL, &prof_event); //extra profiling arguments
		
		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		for (int i = 0; i < tempArray.size(); i++) {
			std::cout << tempArray.at(i) << ' ';
		}
		std::cout << std::endl;

		std::cout << "B = " << B << std::endl;
		std::cout << "local_size = " << local_size << std::endl;

		//display kernel execution time
		std::cout << "Kernel execution time [ns]:" <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		string x;
		std::cin >> x;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	//remove memory for vector from heap
	//delete tempArray;
	//tempArray = NULL;

	return 0;
}

