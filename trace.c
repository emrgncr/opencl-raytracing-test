/*

Today's goal is to visualize a simple triangle with no lighting

*/

// C standard includes
#include <stdio.h>

// OpenCL includes
#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>

#include <png.h>

//-----------------------PNG-----------------------------

void save_png(const char *filename, unsigned char *data, int width,
              int height) {
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
    return;
  }

  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr) {
    fprintf(stderr, "Error: png_create_write_struct failed\n");
    fclose(fp);
    return;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    fprintf(stderr, "Error: png_create_info_struct failed\n");
    png_destroy_write_struct(&png_ptr, NULL);
    fclose(fp);
    return;
  }

  png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  for (int y = 0; y < height; y++) {
    row_pointers[y] =
        (png_bytep)&data[y * width * 4]; // Assuming 4 channels (RGB)
  }

  png_init_io(png_ptr, fp);
  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  // Cleanup
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  free(row_pointers);
}

//------ THE FOLLOWING STOLEN FROM OPENCL EXAMPLES --------

/* Find a GPU or CPU associated with the first available platform

The `platform` structure identifies the first platform identified by the
OpenCL runtime. A platform identifies a vendor's installation, so a system
may have an NVIDIA platform and an AMD platform.

The `device` structure corresponds to the first accessible device
associated with the platform. Because the second parameter is
`CL_DEVICE_TYPE_GPU`, this device must be a GPU.
*/
cl_device_id create_device() {

  cl_platform_id platform;
  cl_device_id dev;
  int err;

  /* Identify a platform */
  err = clGetPlatformIDs(1, &platform, NULL);
  if (err < 0) {
    perror("Couldn't identify a platform");
    exit(1);
  }

  // Access a device
  // GPU
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
  if (err == CL_DEVICE_NOT_FOUND) {
    // CPU
    printf("Cannot find GPU, run on CPU \n");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
  }
  if (err < 0) {
    perror("Couldn't access any devices");
    exit(1);
  }

  return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev,
                         const char *filename) {

  cl_program program;
  FILE *program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;
  int err;

  /* Read program file and place content into buffer */
  program_handle = fopen(filename, "r");
  if (program_handle == NULL) {
    perror("Couldn't find the program file");
    exit(1);
  }
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char *)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  /* Create program from file

  Creates a program from the source code in the add_numbers.cl file.
  Specifically, the code reads the file's content into a char array
  called program_buffer, and then calls clCreateProgramWithSource.
  */
  program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer,
                                      &program_size, &err);
  if (err < 0) {
    perror("Couldn't create the program");
    exit(1);
  }
  free(program_buffer);

  /* Build program

  The fourth parameter accepts options that configure the compilation.
  These are similar to the flags used by gcc. For example, you can
  define a macro with the option -DMACRO=VALUE and turn off optimization
  with -cl-opt-disable.
  */
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err < 0) {

    /* Find size of log and print to std output */
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    program_log = (char *)malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1,
                          program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    exit(1);
  }

  return program;
}

//---------------------------------------------------------

#define CL_RUN(fnc, ...)                                                       \
  fnc(__VA_ARGS__, &err);                                                      \
  if (err < 0) {                                                               \
    printf("CL ERROR AT LINE %d\n", __LINE__);                                 \
    perror("err");                                                             \
    exit(1);                                                                   \
  }

#define CL_RUN2(fnc)                                                           \
  do {                                                                         \
    cl_int err = fnc;                                                          \
    if (err < 0) {                                                             \
      printf("CL ERROR AT LINE %d %d\n", __LINE__, err);                       \
      perror("err");                                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);
//---------------------------------------------------------

#define NUM_PIXELS_X 1920
#define NUM_PIXELS_Y 1080

#define OUT_BUFFER_LEN (NUM_PIXELS_X * NUM_PIXELS_Y)

typedef float float4[4];
typedef float4 triangle[3];
typedef unsigned char uchar;

float *get_in_buffer(triangle *ib, size_t nelem) {
  float *ret = calloc(nelem, sizeof(triangle));
  int i = 0;
  for (int j = 0; j < nelem; j++) {
    for (int m = 0; m < 3; m++) {
      for (int k = 0; k < 3; k++) {
        ret[i++] = ib[j][m][k];
      }
      ret[i++] = 0.0;
    }
  }
  return ret;
}

void translate_traingle(triangle t, float x, float y, float z) {
  for (int i = 0; i < 3; i++) {
    float *point = t[i];
    point[0] += x;
    point[1] += y;
    point[2] += z;
    // printf("%.2f %.2f %.2f \n", point[0], point[1], point[2]);
  }
}

int main() {
  cl_int err;
  cl_device_id device = create_device();

  cl_context context = CL_RUN(clCreateContext, NULL, 1, &device, NULL, NULL);

  cl_program program = build_program(context, device, "trace.cl");

  int num_compute = 4;
  int supersample = 8;

  /*
  use x,y for supersampling,
  use z for object hit buffer
  */

  size_t global_size[3] = {NUM_PIXELS_X * supersample,
                           NUM_PIXELS_Y * supersample, num_compute};
  size_t local_size[3] = {supersample, supersample, num_compute};

  float *out = calloc(OUT_BUFFER_LEN, sizeof(float4));
  if (out == NULL) {
    perror("unable to allocate host pointer");
    exit(1);
  }

  // in buffer contains data regarding the scene geometry
  // for test just have a 3d triangle

  triangle test = {
      {-0.5, -0.5, -0.4, 0.0}, {-.5, .5, -0.4, 0.0}, {.5, 0.0, -1.2, 0.0}};

  translate_traingle(test, -0.6, 0.0, 0.0);
  triangle _in_buffer[] = {
      {{-0.5, -0.5, -1.0, 0.0}, {-.5, 1.5, -1.0, 0.0}, {.5, 0.0, -1.0, 0.0}},
      {}};

  cl_float4 _color_buffer[] = {{1.0f, 0.0f, 0.0f, 1.0f},
                               {0.0f, 1.0f, 0.0f, 1.0f}};

  memcpy(&(_in_buffer[1]), test, sizeof(triangle));
  // translate_traingle(_in_buffer[0], 0.0, 0.0, 0.8);
  // memset(&(_in_buffer[1]), 0, sizeof(triangle));

  int in_size_len = sizeof(_in_buffer) / sizeof(triangle);
  float *in = get_in_buffer(_in_buffer, in_size_len);

  int xsize = NUM_PIXELS_X;
  int ysize = NUM_PIXELS_Y;
  float focal_len = 100.0f;

  cl_mem out_buffer =
      CL_RUN(clCreateBuffer, context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
             OUT_BUFFER_LEN * sizeof(float4), out);

  cl_mem in_buffer =
      CL_RUN(clCreateBuffer, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
             sizeof(_in_buffer), _in_buffer);

  cl_mem color_buffer =
      CL_RUN(clCreateBuffer, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
             sizeof(_color_buffer), _color_buffer);

  cl_queue_properties _properties[] = {CL_QUEUE_PROPERTIES,
                                       CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue queue =
      CL_RUN(clCreateCommandQueueWithProperties, context, device, _properties);

  cl_kernel kernel = CL_RUN(clCreateKernel, program, "trace");

  /**
__kernel void trace(__global float4 *in_data, __global float4 *out_data, __local
float *subpixel_array,
                    __local int *hitbufferi, __local float *hitbuffert,
                    __local float4 *R0buffer, __local float4 *Rdbuffer, __local
float4 *Ribuffer,
                    __local float4 *Nbuffer,
                    int in_data_len, int xsize, int ysize, float focal_len)
  */

  CL_RUN2(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buffer));
  CL_RUN2(clSetKernelArg(kernel, 1, sizeof(cl_mem), &color_buffer));
  CL_RUN2(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buffer));
  CL_RUN2(clSetKernelArg(kernel, 3,
                         sizeof(cl_float4) * supersample * supersample, NULL));
  CL_RUN2(clSetKernelArg(
      kernel, 4, sizeof(cl_int) * supersample * supersample * num_compute,
      NULL));
  CL_RUN2(clSetKernelArg(
      kernel, 5, sizeof(cl_float) * supersample * supersample * num_compute,
      NULL));
  CL_RUN2(clSetKernelArg(kernel, 6,
                         sizeof(cl_float4) * supersample * supersample, NULL));
  CL_RUN2(clSetKernelArg(kernel, 7,
                         sizeof(cl_float4) * supersample * supersample, NULL));
  CL_RUN2(clSetKernelArg(
      kernel, 8, sizeof(cl_float4) * supersample * supersample * num_compute,
      NULL));
  CL_RUN2(clSetKernelArg(
      kernel, 9, sizeof(cl_float4) * supersample * supersample * num_compute,
      NULL));
  CL_RUN2(clSetKernelArg(kernel, 10, sizeof(int), &in_size_len));
  CL_RUN2(clSetKernelArg(kernel, 11, sizeof(int), &xsize));
  CL_RUN2(clSetKernelArg(kernel, 12, sizeof(int), &ysize));
  CL_RUN2(clSetKernelArg(kernel, 13, sizeof(float), &focal_len));

  printf("adad\n");
  CL_RUN2(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size,
                                 local_size, 0, NULL, NULL));
  printf("adad\n");
  CL_RUN2(clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0,
                              OUT_BUFFER_LEN * sizeof(float4), out, 0, NULL,
                              NULL));

  printf("Done with evtr\n");

  int sum = 0;

  for (int i = 0; i < NUM_PIXELS_Y; i++) {
    for (int j = 0; j < NUM_PIXELS_X; j++) {
      sum += out[((i * NUM_PIXELS_X) + j) * 4];
      if (i % 10 == 0 && j % 10 == 0)
        printf("%d ", (int)out[((i * NUM_PIXELS_X) + j) * 4]);
    }
    if (i % 10 == 0)
      printf("\n");
  }
  printf("\nsum: %d\n", sum);

  printf("%.2f ", out[((0 * NUM_PIXELS_X) + 0) * 4]);
  printf("%.2f ", out[((0 * NUM_PIXELS_X) + 1) * 4]);
  printf("%.2f ", out[((0 * NUM_PIXELS_X) + 2) * 4]);

  printf("%.2f\n", out[((250 * NUM_PIXELS_X) + 250) * 4]);

  // convert out to unsigned char
  uchar *outc = calloc(OUT_BUFFER_LEN * 4, sizeof(uchar));
  for (int i = 0; i < OUT_BUFFER_LEN * 4; i++) {
    float f = out[i];
    int k = f * 255.0;
    outc[i] = k;
  }

  save_png("./test.png", outc, NUM_PIXELS_X, NUM_PIXELS_Y);

  clReleaseKernel(kernel);
  clReleaseMemObject(out_buffer);
  clReleaseMemObject(in_buffer);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
  return 0;
}
