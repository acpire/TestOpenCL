// OpenCL.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#pragma comment(lib, "OpenCL.lib")
#pragma comment(lib, "opencv_world400d.lib")
#pragma warning(disable:4996)
cl_float* convertUcharToFloat(cl_uchar* data, size_t length) {
	cl_float* result = (cl_float*)malloc(length * sizeof(cl_float));
	for (size_t i = 0; i < length; i++)
		result[i] = data[i];
	return result;
}
void convertFloatToUchar(cl_float* data, size_t length, cl_uchar* result) {
	if (result)
	for (size_t i = 0; i < length; i++)
		result[i] = data[i];
	free(data);
}
void CL_CALLBACK pfnBuildProgram(cl_program program, void *userData)
{
	size_t numberDevices = *((size_t*)userData);
	cl_device_id *deviceID = (cl_device_id*)((size_t*)userData + 1);
	cl_build_status status;
	size_t length = 0;
	cl_int ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, &length);
	if (status == CL_BUILD_IN_PROGRESS)
	{
		cl_char* data = NULL;
		for (size_t i = 0; i < numberDevices; i++) {
			cl_int ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &length);
			if (ret != CL_SUCCESS)
				printf("clGetProgramBuildInfo - %d", ret);
			data = (cl_char*)realloc(data, length + 1);
			data[length] = 0;
			ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, length, data, NULL);
			if (ret != CL_SUCCESS)
				printf("clGetProgramBuildInfo - %d", ret);
			printf("CL_BUILD_IN_PROGRESS - %s\n", data);
		}
		if (data)
			free(data);
	}
	else if (status == CL_BUILD_ERROR)
	{
		cl_char* data = NULL;
		for (size_t i = 0; i < numberDevices; i++) {
			cl_int ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &length);
			if (ret != CL_SUCCESS)
				printf("clGetProgramBuildInfo - %d", ret);
			data = (cl_char*)realloc(data, length + 1);
			data[length] = 0;
			ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, length, data, NULL);
			if (ret != CL_SUCCESS)
				printf("clGetProgramBuildInfo - %d", ret);
			printf("CL_BUILD_ERROR - %s", data);
		}
		if (data)
			free(data);
	}
	else if (status == CL_BUILD_NONE) {
		printf("CL_BUILD_NONE");
	}
	else if (status == CL_BUILD_SUCCESS) {
		printf("CL_BUILD_IN_PROGRESS");
	}
}
#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "\nOpenCL Error: '%s' returned %d!\n", #_expr, (cl_int)_err);   \
     abort();                                                                   \
    } while (0)

char* loadFile(const char* filename)
{
	size_t lengthFile;
	char* data;
	FILE* kernel;
	fopen_s(&kernel, filename, "rb");
	if (kernel == NULL)
		return NULL;
	fseek(kernel, 0, SEEK_END);
	lengthFile = ftell(kernel);
	fseek(kernel, 0, SEEK_SET);
	data = (char*)malloc(lengthFile + 1);
	size_t result = fread(data, lengthFile, 1, kernel);
	fclose(kernel);

	if (result != 1) {
		free(data);
		return NULL;
	}

	data[lengthFile] = '\0';
	printf("%s\n", data);
	return data;
}

void platformInfo(cl_platform_id platform, cl_platform_info information, const char textInformation[]) {
	size_t length;
	clGetPlatformInfo(platform, information, NULL, NULL, &length);
	char* data = (char*)_malloca(length + 1);
	data[length] = 0;
	clGetPlatformInfo(platform, information, length, data, NULL);
	printf("%s - %s\n", textInformation, data);
	_freea(data);
}
void deviceInfo(cl_device_id device, cl_platform_info information, const char textInformation[]) {
	size_t length;
	clGetDeviceInfo(device, information, NULL, NULL, &length);
	cl_ulong data;
	clGetDeviceInfo(device, information, length, &data, NULL);
	printf("%s - %llu\n", textInformation, data);
}
void buildInfo(cl_device_id device, cl_program program, cl_program_build_info info, const char textInformation[]) {
	size_t length;
	clGetProgramBuildInfo(program, device, info, NULL, NULL, &length);
	char* data = (char*)_malloca(length + 1);
	data[length] = 0;
	clGetProgramBuildInfo(program, device, info, length, data, NULL);
	printf("%s - %s\n", textInformation, data);
	_freea(data);
}

bool InitOpenCL(cl_uint& numberPlatforms, cl_uint*& numberDevices, cl_platform_id*& platforms,
	cl_program*& programDevices, cl_context*& context, cl_device_id**& devices,
	cl_command_queue**& queue, cl_kernel***& kernels) {
	const char* code = loadFile("Code.cl");
	cl_int errorCode;

	CL_CHECK(clGetPlatformIDs(NULL, NULL, &numberPlatforms));

	platforms = (cl_platform_id*)malloc(numberPlatforms * sizeof(cl_platform_id));
	numberDevices = (cl_uint*)malloc(numberPlatforms * sizeof(cl_uint));
	devices = (cl_device_id**)malloc(numberPlatforms * sizeof(cl_device_id*));
	queue = (cl_command_queue**)malloc(numberPlatforms * sizeof(cl_command_queue*));
	context = (cl_context*)malloc(numberPlatforms * sizeof(cl_context));
	programDevices = (cl_program*)malloc(numberPlatforms * sizeof(cl_program));
	kernels = (cl_kernel***)malloc(numberPlatforms * sizeof(cl_kernel**));

	CL_CHECK(clGetPlatformIDs(numberPlatforms, platforms, NULL));

	for (size_t i = 0; i < numberPlatforms; i++) {
		CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, NULL, NULL, &numberDevices[i]);
		devices[i] = (cl_device_id*)malloc(numberDevices[i] * sizeof(cl_device_id)));
		kernels[i] = (cl_kernel**)malloc(numberDevices[i] * sizeof(cl_kernel*));
		queue[i] = (cl_command_queue*)malloc(numberDevices[i] * sizeof(cl_command_queue));
		CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numberDevices[i], devices[i], NULL));
	}
	for (size_t i = 0; i < numberPlatforms; i++) {
		platformInfo(platforms[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
		platformInfo(platforms[i], CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
		platformInfo(platforms[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
		platformInfo(platforms[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
		platformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
		for (size_t j = 0; j < numberDevices[i]; j++) {
			deviceInfo(devices[i][j], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
			deviceInfo(devices[i][j], CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
			deviceInfo(devices[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
			deviceInfo(devices[i][j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
			deviceInfo(devices[i][j], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
		}
	}

	for (size_t i = 0; i < numberPlatforms; i++) {
		cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0 };
		context[i] = clCreateContext(properties, numberDevices[i], devices[i], NULL, NULL, &errorCode);
		CL_CHECK(errorCode);
	}
	for (size_t i = 0; i < numberPlatforms; i++) {
		//cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
		for (size_t j = 0; j < numberDevices[i]; j++) {
			queue[i][j] = clCreateCommandQueue(context[i], devices[i][j], CL_QUEUE_PROFILING_ENABLE, &errorCode);
			CL_CHECK(errorCode);
		}
	}
	
	for (size_t i = 0; i < numberPlatforms; i++) {
		programDevices[i] = clCreateProgramWithSource(context[i], 1, &code, NULL, &errorCode);
		CL_CHECK(errorCode);
		size_t* data = (size_t*)malloc(sizeof(size_t) + numberDevices[i] * sizeof(cl_device_id));
		if (numberDevices[i] != 0) {
			data[0] = numberDevices[i];
			memcpy(&data[1], devices[i], data[0] * sizeof(cl_device_id));
		}

		CL_CHECK(clBuildProgram(programDevices[i], numberDevices[i], devices[i], "", pfnBuildProgram, (void*)data));

		free(data);
		for (size_t j = 0; j < numberDevices[i]; j++) {
			//CL_CHECK(clBuildProgram(programDevices[i], 1, &devices[i][j], "", NULL, NULL));

			kernels[i][j] = (cl_kernel*)malloc(2 * sizeof(cl_kernel));
			kernels[i][j][0] = clCreateKernel(programDevices[i], "convolution_global", &errorCode);
			//kernels[i][j][1] = clCreateKernel(programDevices[i], "my_filter", &errorCode);
			CL_CHECK(errorCode);
			
		}
	}
	return true;
}

bool convolutionBufferCalc(cl_uchar* data, cl_uint width, cl_uint height,
	cl_uchar* kernel, cl_uint widthKernel, cl_uint heightKernel,
	cl_context& context, cl_command_queue& queue, cl_device_id& device,
	cl_kernel& kernelDevice) {

	cl_bool imageSupport;
	CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL));
	if (!imageSupport)
		return false;

	cl_event kernelEvent;
	cl_ulong time_start, time_end;
	cl_int errors;
	cl_image_format imageFormat;
	cl_float div = 1.0f;
	cl_float bias = 0.0f;
	const size_t imageSize[3] = { width, height, 1 };
	const size_t offset[3] = { 0, 0, 0 };
	const size_t global[] = { width , height };
	const size_t local[2] = { 16 , 16 };
	imageFormat.image_channel_order = CL_RGBA;
	imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

	cl_mem firstImage = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(cl_char4) * width * height, NULL, &errors);
	CL_CHECK(errors);

	cl_mem secondImage = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(char) * widthKernel * heightKernel, NULL, &errors);
	CL_CHECK(errors);

	CL_CHECK(clEnqueueWriteBuffer(queue, firstImage, CL_FALSE, 0, sizeof(cl_char4) * width * height, data, NULL, NULL, NULL));

	CL_CHECK(clEnqueueWriteBuffer(queue, secondImage, CL_FALSE, 0, sizeof(char) * widthKernel * heightKernel, kernel, NULL, NULL, NULL));
	//CL_CHECK(clEnqueueWriteImage(queue, firstImage, CL_FALSE, offset, imageSize, NULL, NULL, data, NULL, NULL, NULL));;
	CL_CHECK(clSetKernelArg(kernelDevice, 0, sizeof(firstImage), &firstImage));
	CL_CHECK(clSetKernelArg(kernelDevice, 1, sizeof(secondImage), &secondImage));
	CL_CHECK(clSetKernelArg(kernelDevice, 2, sizeof(widthKernel), &widthKernel));
	CL_CHECK(clSetKernelArg(kernelDevice, 3, sizeof(width), &width));
	//CL_CHECK(clSetKernelArg(kernelDevice, 4, sizeof(height), &height));
	//CL_CHECK(clSetKernelArg(kernelDevice, 4, sizeof(div), &div));
	//CL_CHECK(clSetKernelArg(kernelDevice, 5, sizeof(bias), &bias));

	{
		cl_ulong local_mem_size;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL));
		printf("CL KERNEL LOCAL MEM SIZE: %llu\n", local_mem_size);
		cl_ulong private_mem_size;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &private_mem_size, NULL));
		printf("CL KERNEL PRIVATE MEM SIZE: %llu\n", private_mem_size);
		size_t work_group_size;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL));
		printf("CL KERNEL WORK GROUP SIZE: %zu\n", work_group_size);
		size_t preferred_work_group_size_multiple;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_work_group_size_multiple, NULL));
		printf("CL KERNEL PREFERRED WORK GROUP SIZE MULTIPLE: %zu\n", preferred_work_group_size_multiple);
		size_t global_work_size[3];
		clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(global_work_size), global_work_size, NULL);
		printf("CL KERNEL GLOBAL WORK SIZE: %zu, %zu, %zu\n", global_work_size[0], global_work_size[1], global_work_size[2]);
		size_t compile_work_group_size[3];
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(compile_work_group_size), compile_work_group_size, NULL));
		printf("CL KERNEL COMPILE WORK GROUP SIZE: %zu, %zu, %zu\n", compile_work_group_size[0], compile_work_group_size[1], compile_work_group_size[2]);
		cl_uint vec_width[6];
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_int), &vec_width[0], NULL);
		clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &vec_width[1], NULL);
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_int), &vec_width[2], NULL);
		clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(cl_int), &vec_width[3], NULL);
		clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_int), &vec_width[4], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_int), &vec_width[5], NULL);
		printf("LOCAL_MEMORY %u\n"\
			"COMPUTE_UNITS %u\n"\
			"MAX_WORK_GROUP_SIZE %u\n"\
			"NATIVE_VECTOR_WIDTH_FLOAT %u\n"\
			"SINGLE_FP_CONFIG %u\n"\
			"PREFERRED_VECTOR_WIDTH_FLOAT %u\n\n", vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4], vec_width[5]);
	}
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernelDevice, 2, NULL, global, NULL, NULL, NULL, &kernelEvent));
	CL_CHECK(clEnqueueReadBuffer(queue, firstImage, CL_TRUE, 0, sizeof(cl_char4) * width * height, data, NULL, NULL, NULL));

	//CL_CHECK(clEnqueueReadImage(queue, resultImage, CL_TRUE, offset, imageSize, NULL, NULL, data, NULL, NULL, &kernelEvent));;
	CL_CHECK(clWaitForEvents(1, &kernelEvent));

	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL));
	const cl_ulong total_time = time_end - time_start;
	const cl_double time = total_time / 1000000.0;
	printf("Execution time:\t\t%0f ms\n", time);

	CL_CHECK(clReleaseMemObject(firstImage));
	CL_CHECK(clReleaseMemObject(secondImage));
	return true;
}

bool convolutionCalc(cl_uchar* data, size_t width, size_t height,
	cl_uchar* kernel, cl_int widthKernel, size_t heightKernel,
	cl_context& context, cl_command_queue& queue, cl_device_id& device,
	cl_kernel& kernelDevice) {
	cl_float* dataImage = convertUcharToFloat(data, width * height * 4);
	cl_float* dataKernel = convertUcharToFloat(kernel, widthKernel * heightKernel);
	cl_bool imageSupport;
	CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL));
	if (!imageSupport)
		return false;

	cl_event kernelEvent;
	cl_ulong time_start, time_end;
	cl_int errors;
	cl_image_format imageFormat;
	cl_float div = 0.1f;
	cl_float bias = 0.0f;
	const size_t imageSize[3] = { width, height, 1 };
	const size_t offset[3] = { 0, 0, 0 };
	const size_t global[2] = { 4 * width , height };
	const size_t local[2] = { 16 , 16 };
	imageFormat.image_channel_order = CL_LUMINANCE;
	imageFormat.image_channel_data_type = CL_FLOAT;

	cl_image_desc image_desc = { 0 };
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = width * 4;
	image_desc.image_height = height;
	image_desc.image_array_size = 0;
//	image_desc.image_row_pitch = width * sizeof(cl_float4);
//	image_desc.image_slice_pitch = width * sizeof(cl_float4) * height;
	image_desc.num_mip_levels = 0;
	image_desc.num_samples = 0;
	image_desc.buffer = NULL;

	//cl_mem firstImage = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	//	&imageFormat, width, height, &image_desc, dataImage, &errors);
	//CL_CHECK(errors);
	//cl_mem resultImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY,
	//	&imageFormat, width, height, &image_desc, NULL, &errors);
	//CL_CHECK(errors);

	cl_mem firstImage = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &imageFormat, &image_desc, dataImage, &errors);
	CL_CHECK(errors);
	cl_mem resultImage = clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &imageFormat, &image_desc, dataImage, &errors);
	CL_CHECK(errors);
	cl_mem secondImage = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * widthKernel * heightKernel, NULL, &errors);
	CL_CHECK(errors);

	CL_CHECK(clEnqueueWriteBuffer(queue, secondImage, CL_FALSE, 0, sizeof(cl_float) * widthKernel * heightKernel, dataKernel, NULL, NULL, NULL));
	//CL_CHECK(clEnqueueWriteImage(queue, firstImage, CL_FALSE, offset, imageSize, NULL, NULL, data, NULL, NULL, NULL));;
	CL_CHECK(clSetKernelArg(kernelDevice, 0, sizeof(resultImage), &resultImage));
	CL_CHECK(clSetKernelArg(kernelDevice, 1, sizeof(firstImage), &firstImage));
	CL_CHECK(clSetKernelArg(kernelDevice, 2, sizeof(widthKernel), &widthKernel));
	CL_CHECK(clSetKernelArg(kernelDevice, 3, sizeof(secondImage), &secondImage));
	CL_CHECK(clSetKernelArg(kernelDevice, 4, sizeof(div), &div));
	CL_CHECK(clSetKernelArg(kernelDevice, 5, sizeof(bias), &bias));

	{
		cl_ulong local_mem_size;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL));
		printf("CL KERNEL LOCAL MEM SIZE: %llu\n", local_mem_size);
		cl_ulong private_mem_size;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &private_mem_size, NULL));
		printf("CL KERNEL PRIVATE MEM SIZE: %llu\n", private_mem_size);
		size_t work_group_size;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL));
		printf("CL KERNEL WORK GROUP SIZE: %zu\n", work_group_size);
		size_t preferred_work_group_size_multiple;
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_work_group_size_multiple, NULL));
		printf("CL KERNEL PREFERRED WORK GROUP SIZE MULTIPLE: %zu\n", preferred_work_group_size_multiple);
		size_t global_work_size[3];
		clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(global_work_size), global_work_size, NULL);
		printf("CL KERNEL GLOBAL WORK SIZE: %zu, %zu, %zu\n", global_work_size[0], global_work_size[1], global_work_size[2]);
		size_t compile_work_group_size[3];
		(clGetKernelWorkGroupInfo(kernelDevice, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(compile_work_group_size), compile_work_group_size, NULL));
		printf("CL KERNEL COMPILE WORK GROUP SIZE: %zu, %zu, %zu\n", compile_work_group_size[0], compile_work_group_size[1], compile_work_group_size[2]);
		cl_uint vec_width[6];
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_int), &vec_width[0], NULL);
		clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &vec_width[1], NULL);
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_int), &vec_width[2], NULL);
		clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(cl_int), &vec_width[3], NULL);
		clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_int), &vec_width[4], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_int), &vec_width[5], NULL);
		printf("LOCAL_MEMORY %u\n"\
			"COMPUTE_UNITS %u\n"\
			"MAX_WORK_GROUP_SIZE %u\n"\
			"NATIVE_VECTOR_WIDTH_FLOAT %u\n"\
			"SINGLE_FP_CONFIG %u\n"\
			"PREFERRED_VECTOR_WIDTH_FLOAT %u\n\n", vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4], vec_width[5]);
	}
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernelDevice, 2, NULL, global, NULL, NULL, NULL, &kernelEvent));
	clFinish(queue);
	//CL_CHECK(clEnqueueReadImage(queue, resultImage, CL_TRUE, offset, imageSize, NULL, NULL, dataImage, NULL, NULL, &kernelEvent));;
	CL_CHECK(clWaitForEvents(1, &kernelEvent));

	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL));
	const cl_ulong total_time = time_end - time_start;
	const cl_double time = total_time / 1000000.0;
	printf("Execution time:\t\t%0f ms\n", time);

	CL_CHECK(clReleaseMemObject(firstImage));
	CL_CHECK(clReleaseMemObject(secondImage));
	CL_CHECK(clReleaseMemObject(resultImage));
	convertFloatToUchar(dataImage, width*height * 4, data);
	convertFloatToUchar(dataKernel, widthKernel*heightKernel, NULL);
	return true;
}
int main()
{
	cv::Mat image = cv::imread("C:/Users/human/source/repos/OpenCL/x64/Debug/image.jpg", cv::IMREAD_ANYCOLOR);
	cv::Mat image2(image.rows, image.cols, CV_8UC4, cv::Scalar(0.0f));
	cl_uchar* dataImage2 = (cl_uchar*)image2.data;
	cl_uchar* dataImage = (cl_uchar*)image.data;
	for (size_t i = 0; i < image.rows * image.cols; i++) {
		dataImage2[i * 4] = dataImage[i * 3];
		dataImage2[i * 4 + 1] = dataImage[i * 3 + 1];
		dataImage2[i * 4 + 2] = dataImage[i * 3 + 2];
		dataImage2[i * 4 + 3] = 0;
	}
	cv::namedWindow("My Image", cv::WINDOW_NORMAL);
	//cv::resizeWindow("My Image", cv::Size(1280, 720));
	cv::imshow("My Image", image2);
	cv::waitKey(0);

	uint8_t kernel[9] = { 1,0,1,
		2,0,2,
		1,0,1 };
	cl_uint numberPlatforms;
	cl_uint* numberDevices;
	cl_platform_id* platforms;
	cl_program* programDevices;
	cl_context* context;
	cl_device_id** devices;
	cl_command_queue** queue;
	cl_kernel*** kernels;
	InitOpenCL(numberPlatforms, numberDevices, platforms, programDevices, context, devices, queue, kernels);
	//float_t div = 0.5f;
	for (size_t i = 0; i < numberPlatforms; i++) {
		for (size_t j = 2; j < numberDevices[i]; j++) {
			//CL_CHECK(clSetKernelArg(kernels[i][j][0], 4, sizeof(div), (void*)&div));
			convolutionCalc(image2.data, image2.size().width, image2.size().height,
				(uint8_t*)kernel, 3, 3, context[i], queue[i][j], devices[i][j], kernels[i][j][0]);
			cv::namedWindow("My Image", cv::WINDOW_AUTOSIZE);
			//cv::resizeWindow("My Image", cv::Size(1280, 720));
			cv::imshow("My Image", image2);
			cv::imwrite("result.jpg", image2);
		}
	}
			cv::waitKey(0);
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.