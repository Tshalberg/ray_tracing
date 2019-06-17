#include <iostream>
#include <time.h>
#include "float.h"

#include <curand_kernel.h>
#include "stdio.h"

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"

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
vec3 random_in_unit_sphere(curandState *local_rand_state) {
	vec3 p;
	do {
		p = 2.0f*vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1,1,1);
	} while (p.squared_length() >= 1.0f);
	return p;
}



__device__ 
vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	// printf("color cur_ray.direction: %f, %f, %f \n", cur_ray.direction().r(), cur_ray.direction().g(), cur_ray.direction().b());
	// printf("color cur_ray.origin: %f, %f, %f \n", cur_ray.origin().r(), cur_ray.origin().g(), cur_ray.origin().b());
	float cur_attenuation = 1.0f;
	for (int i = 0; i<50; i++){
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)){
			vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
			cur_attenuation *= 0.5f;
			cur_ray = ray(rec.p, target-rec.p);
			// if (i == 0){
			// 	printf("color cur_ray.direction: %f, %f, %f \n", cur_ray.direction().r(), cur_ray.direction().g(), cur_ray.direction().b());
			// 	printf("color cur_ray.origin: %f, %f, %f \n", cur_ray.origin().r(), cur_ray.origin().g(), cur_ray.origin().b());
			// }
		}
		else {
			// printf(": %f, %f \n", , );
			// vec3 randVec = random_in_unit_sphere(local_rand_state);
			// printf("color random_in_unit_sphere: %f, %f, %f \n", randVec.r(), randVec.g(), randVec.b());
			// printf("color cur_ray.direction: %f, %f, %f \n", cur_ray.direction().r(), cur_ray.direction().g(), cur_ray.direction().b());
			// printf("color cur_ray.origin: %f, %f, %f \n", cur_ray.origin().r(), cur_ray.origin().g(), cur_ray.origin().b());
			vec3 unit_direction = unit_vector(cur_ray.direction());
			// printf("color unit_direction: %f, %f, %f \n", unit_direction.r(), unit_direction.g(), unit_direction.b());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			// printf("color t: %f \n", t);
			vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
			// printf("color c: %f, %f, %f \n", c.r(), c.g(), c.b());
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__
void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	delete *(d_list);
	delete *(d_list+1);
	delete *d_world;
	delete *d_camera;
}


__global__
void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list)   = new sphere(vec3(0, 0, -1), 0.5);
		*(d_list+1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world    = new hitable_list(d_list,2);
		*d_camera   = new camera();
	}
}


__global__
void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


__global__
void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s=0; s < ns; s++){
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		// printf("render u, v: %f, %f \n", u, v);
		ray r = (*cam)->get_ray(u,v);
		// printf("render r.direction: %f, %f, %f \n", r.direction().r(), r.direction().g(), r.direction().b());
		// printf("render r.origin: %f, %f, %f \n", r.origin().r(), r.origin().g(), r.origin().b());
		col += color(r, world, &local_rand_state);
		// printf("render(in loop): %f, %f, %f \n", col.r(), col.g(), col.b());
	}
	// printf("render: %f, %f, %f \n", col.r(), col.g(), col.b());
	col /= float(ns);
	fb[pixel_index] = col;
}


int main() {
	int nx = 1920;
	int ny = 1080;
	// int nx = 600;
	// int ny = 300;
	int ns = 100;
	int num_pixels = nx*ny;
	size_t fb_size = num_pixels*sizeof(vec3);

	// allocate FB
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)& fb, fb_size));

	int tx = 8;
	int ty = 8;

	// Creating random states for GPU
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)& d_rand_state, num_pixels*sizeof(curandState)));

	// Creating world
	hitable **d_list;
	checkCudaErrors(cudaMalloc((void **)& d_list, 2*sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)& d_world, sizeof(hitable *)));
	// Creating camera
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)& d_camera, sizeof(camera *)));

	create_world<<<1,1>>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Start timing
    clock_t start, stop;
    start = clock();

	// Render our buffer
	dim3 blocks(nx/tx+1, ny/ty+1);
	dim3 threads(tx, ty);

	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
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
			// std::cout << r << " " << g << " " << b << "\n";
			int ir = int(255.99*r);
			int ig = int(255.99*g);
			int ib = int(255.99*b);
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}
	// Free memory
	checkCudaErrors(cudaDeviceSynchronize());
	free_world<<<1, 1>>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(fb));
}