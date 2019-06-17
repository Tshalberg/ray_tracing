#include <iostream>
#include <time.h>

#include "vec3.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ 
vec3 color(const ray& r) {
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f*(unit_direction.y() + 1.0f);
	return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0); //t*vec3(0.2, 1.0, 0.2);//t*vec3(0.5, 0.7, 1.0);
}


__global__
void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u*horizontal + v*vertical);
	fb[pixel_index] = color(r);
}


int main() {
	int nx = 1920;
	int ny = 1080;
	int num_pixels = nx*ny;
	size_t fb_size = num_pixels*sizeof(vec3);

	// allocate FB
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	int tx = 8;
	int ty = 8;

	// Define lower_left_corner, horizontal and vertical vectors
	vec3 lower_left_corner(-2.0, -1.0, -1.0);
	vec3 horizontal(4.0, 0.0, 0.0);
	vec3 vertical(0.0, 2.0, 0.0);
	vec3 origin(0.0, 0.0, 0.0);

	// Start timing
    clock_t start, stop;
    start = clock();

	// Render our buffer
	dim3 blocks(nx/tx+1, ny/ty+1);
	dim3 threads(tx, ty);
	render<<<blocks, threads>>>(fb, nx, ny, lower_left_corner, horizontal, vertical, origin);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Stop timing and print out result
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

	// Output FB as Image
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny-1; j>=0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j*nx + i;
			float r = fb[pixel_index].r();
			float g = fb[pixel_index].g();
			float b = fb[pixel_index].b();
			int ir = int(255.99*r);
			int ig = int(255.99*g);
			int ib = int(255.99*b);
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}
	checkCudaErrors(cudaFree(fb));
}