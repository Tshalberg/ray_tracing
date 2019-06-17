#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

// vec3 random_in_unit_disc() {
// 	vec3 p;
// 	do {
// 		p = 2.0*vec3(drand48(), drand48(), 0) - vec3(1,1,0);
// 	} while (dot(p,p) >= 1);
// 	return p;
// }

class camera {
	public:
		__device__ camera() { // vfov is top to bottom in degrees
			// Define lower_left_corner, horizontal and vertical vectors
			lower_left_corner = vec3(-2.0, -1.0, -1.0);
			horizontal = vec3(4.0, 0.0, 0.0);
			vertical = vec3(0.0, 2.0, 0.0);
			origin = vec3(0.0, 0.0, 0.0);
		}
		__device__ ray get_ray(float u, float v) {
			// printf("camera u, v: %f, %f \n", u, v);
			vec3 direction = lower_left_corner + u*horizontal + v*vertical - origin;
			// printf("get_ray direction: %f, %f, %f \n", direction.r(), direction.g(), direction.b());
			return ray(origin, direction); }

		vec3 origin;
		vec3 lower_left_corner;
		vec3 horizontal;
		vec3 vertical;

};

#endif