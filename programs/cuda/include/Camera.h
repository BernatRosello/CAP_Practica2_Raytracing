#pragma once

//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#include "Vec3.h"
#include "ray.h"
#include "random.h"

#include <curand_kernel.h>

class Camera {
public:
    __host__ __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
        // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*float(PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2*half_width*focus_dist*u;
        vertical = 2*half_height*focus_dist*v;
    }
    __host__ Ray get_ray(float s, float t) {
        Vec3 rd = lens_radius*randomNormalDisk();
        Vec3 offset = u * rd.x() + v * rd.y();
        return Ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }

    __device__ Ray get_ray(float s, float t, curandState* local_rand_state) {
        Vec3 rd = lens_radius * randomNormalDiskGPU(local_rand_state);
        Vec3 offset = u * rd.x() + v * rd.y();
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }


private:
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;

	const double PI = 3.1415926535897;

    __device__ Vec3 randomNormalDiskGPU(curandState* local_rand_state) {
        Vec3 p(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0);
        while (dot(p, p) >= 1.0f) {
            p = 2.0f * Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vec3(1, 1, 0);
        }
        return p;
    }
};
