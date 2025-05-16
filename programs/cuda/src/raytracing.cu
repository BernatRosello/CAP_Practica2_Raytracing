// #include <float.h>
#include <cstdio>
// #include <cstdlib>
// #include <limits>

#include <ctime>

#include "Vec3.h"
#include "Ray.h"
#include "Camera.h"
// #include "Object.h"
#include "Scene.h"
#include "Sphere.h"
#include "Diffuse.h"
#include "Metallic.h"
#include "Crystalline.h"

#include "random.h"
// #include "utils.h"
#include "MaterialData.h"
#include "device_launch_parameters.h"
#include <string>
#include <fstream>
#include <sstream>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int w, int h, curandState *rand_state)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if ((idx >= w) || (idy >= h))
        return;

    int pixel_index = idy * w + idx;
    curand_init(42, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int w, int h, int ns, Camera **cam, SceneGPU *world, curandState *rand_state)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if ((idx >= w) || (idy >= h))
        return;

    int pixel_index = idy * w + idx;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++)
    {
        float u = float(idx + curand_uniform(&local_rand_state)) / float(w);
        float v = float(idy + curand_uniform(&local_rand_state)) / float(h);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += world->getSceneColor(r, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Object **aux, int numobjects, SceneGPU *d_world, Camera **d_camera, int nx, int ny, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        d_world->setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
        d_world->setInfColor(Vec3(1.0f, 1.0f, 1.0f));
        d_world->setList(aux, numobjects);
        d_world->add(new Object(
            new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f),
            new Diffuse(Vec3(0.5f, 0.5f, 0.5f))));
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = RND;
                Vec3 center(a + RND, 0.2f, b + RND);
                if (choose_mat < 0.8f)
                {
                    d_world->add(new Object(
                        new Sphere(center, 0.2f),
                        new Diffuse(Vec3(RND * RND,
                                         RND * RND,
                                         RND * RND))));
                }
                else if (choose_mat < 0.95f)
                {
                    d_world->add(new Object(
                        new Sphere(center, 0.2f),
                        new Metallic(Vec3(0.5f * (1.0f + RND),
                                          0.5f * (1.0f + RND),
                                          0.5f * (1.0f + RND)),
                                     0.5f * RND)));
                }
                else
                {
                    d_world->add(new Object(
                        new Sphere(center, 0.2f),
                        new Crystalline(1.5f)));
                }
            }
        }
        d_world->add(new Object(
            new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f),
            new Crystalline(1.5f)));
        d_world->add(new Object(
            new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f),
            new Diffuse(Vec3(0.4f, 0.2f, 0.1f))));
        d_world->add(new Object(
            new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f),
            new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)));

        *rand_state = local_rand_state;

        Vec3 lookfrom(13.0f, 2.0f, 3.0f);
        Vec3 lookat(0.0f, 0.0f, 0.0f);
        float dist_to_focus = 10.0f; //(lookfrom - lookat).length();
        float aperture = 0.1f;
        *d_camera = new Camera(lookfrom,
                               lookat,
                               Vec3(0.0f, 1.0f, 0.0f),
                               20.0,
                               float(nx) / float(ny),
                               aperture,
                               dist_to_focus);
    }
}

// Kernel to build scene objects from SphereData
__global__ void build_scene_from_data(SphereData* data, int numobjects, SceneGPU* d_world, Object** d_objects)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numobjects) return;

    SphereData s = data[idx];
    Vec3 center(s.center[0], s.center[1], s.center[2]);
    float radius = s.radius;
    MaterialData m = s.material;

    Material* mat = nullptr;
    switch (m.type) {
    case DIFFUSE:
        mat = new Diffuse(Vec3(m.color[0], m.color[1], m.color[2]));
        break;
    case METALLIC:
        mat = new Metallic(Vec3(m.color[0], m.color[1], m.color[2]), m.mat_property);
        break;
    case CRYSTALLINE:
        mat = new Crystalline(m.mat_property);
        break;
    default:
        mat = new Diffuse(Vec3(0.5f, 0.5f, 0.5f)); // fallback
    }

    d_objects[idx] = new Object(new Sphere(center, radius), mat);
}

// Kernel to create camera (run once)
__global__ void create_camera(Camera** d_camera, int nx, int ny)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Vec3 lookfrom(13.0f, 2.0f, 3.0f);
        Vec3 lookat(0.0f, 0.0f, 0.0f);
        float dist_to_focus = 10.0f; //(lookfrom - lookat).length();
        float aperture = 0.1f;
        *d_camera = new Camera(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx) / float(ny), aperture, dist_to_focus);
    }
}

void loadSceneToGPUFromFile(const std::string& filename, SceneGPU* d_world, Object*** d_object_list_ptr, int* num_objects_out, Camera** d_camera, int nx, int ny, curandState* rand_state)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<SphereData> sphereDataList;

    if (!file.is_open()) {
        std::cerr << "Failed to open scene file: " << filename << std::endl;
        exit(1);
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (ss >> token) tokens.push_back(token);
        if (tokens.empty()) continue;

        if (tokens[0] == "Object" && tokens[1] == "Sphere") {
            // Example line format:
            // Object Sphere ( x y z r ) MaterialType params...
            try {
                // Expect tokens like: Object Sphere ( 1.0 2.0 3.0 0.5 ) Diffuse 0.7 0.6 0.5
                // Find positions of '(' and ')'
                size_t openParenPos = line.find('(');
                size_t closeParenPos = line.find(')');
                if (openParenPos == std::string::npos || closeParenPos == std::string::npos) {
                    std::cerr << "Malformed sphere line: " << line << std::endl;
                    continue;
                }
                std::string coords_str = line.substr(openParenPos + 1, closeParenPos - openParenPos - 1);
                std::stringstream coords_ss(coords_str);
                float sx, sy, sz, sr;
                coords_ss >> sx >> sy >> sz >> sr;

                SphereData sphere;
                sphere.center[0] = sx;
                sphere.center[1] = sy;
                sphere.center[2] = sz;
                sphere.radius = sr;

                std::string matType;
                std::stringstream mat_ss(line.substr(closeParenPos + 1));
                mat_ss >> matType;

                if (matType == "Crystalline") {
                    sphere.material.type = CRYSTALLINE;
                    sphere.material.color[0] = 0.0f;
                    sphere.material.color[1] = 0.0f;
                    sphere.material.color[2] = 0.0f;
                    float refr_idx;
                    mat_ss >> refr_idx;
                    sphere.material.mat_property = refr_idx;
                }
                else if (matType == "Metallic") {
                    sphere.material.type = METALLIC;
                    float r, g, b, fuzz;
                    mat_ss >> r >> g >> b >> fuzz;
                    sphere.material.color[0] = r;
                    sphere.material.color[1] = g;
                    sphere.material.color[2] = b;
                    sphere.material.mat_property = fuzz;
                }
                else if (matType == "Diffuse") {
                    sphere.material.type = DIFFUSE;
                    float r, g, b;
                    mat_ss >> r >> g >> b;
                    sphere.material.color[0] = r;
                    sphere.material.color[1] = g;
                    sphere.material.color[2] = b;
                    sphere.material.mat_property = 0.0f;
                }
                else {
                    std::cerr << "Unknown material: " << matType << " in line: " << line << std::endl;
                    continue;
                }

                sphereDataList.push_back(sphere);
            }
            catch (...) {
                std::cerr << "Parsing error in line: " << line << std::endl;
                continue;
            }
        }
    }

    file.close();

    int numObjects = static_cast<int>(sphereDataList.size());
    *num_objects_out = numObjects;

    // 1. Allocate device memory for SphereData[]
    SphereData* d_sphere_data = nullptr;
    checkCudaErrors(cudaMalloc(&d_sphere_data, sizeof(SphereData) * numObjects));
    checkCudaErrors(cudaMemcpy(d_sphere_data, sphereDataList.data(), sizeof(SphereData) * numObjects, cudaMemcpyHostToDevice));

    // 2. Allocate device memory for Object* array
    Object** d_objects = nullptr;
    checkCudaErrors(cudaMalloc(&d_objects, sizeof(Object*) * numObjects));
    *d_object_list_ptr = d_objects;

    // 3. Initialize SceneGPU on host, then copy to device
    SceneGPU h_scene;
    h_scene.ol = d_objects;       // device pointer to objects
    h_scene.capacity = numObjects;
    h_scene.size = numObjects;
    h_scene.sky = Vec3(0.5f, 0.7f, 1.0f);
    h_scene.inf = Vec3(1.0f, 1.0f, 1.0f);

    checkCudaErrors(cudaMemcpy(d_world, &h_scene, sizeof(SceneGPU), cudaMemcpyHostToDevice));

    // 4. Launch kernel to build the objects in parallel
    int threads = 256;
    int blocks = (numObjects + threads - 1) / threads;
    build_scene_from_data << <blocks, threads >> > (d_sphere_data, numObjects, d_world, d_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // 5. Create camera (once)
    create_camera << <1, 1 >> > (d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // 6. Free temporary SphereData array
    checkCudaErrors(cudaFree(d_sphere_data));
}

void rayTracingGPU(Vec3* img, int w, int h, int ns = 1)
{
    int tx = 8;
    int ty = 8;
    clock_t start, stop;
    double timer_seconds;

    std::cerr << "Rendering a " << w << "x" << h << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\






        ChatGPT said :
    n";

        arduino
        Copy
        Edit
        int num_pixels = w * h;
    Vec3* fb;
    checkCudaErrors(cudaMallocManaged(&fb, num_pixels * sizeof(Vec3)));

    Camera* d_camera = nullptr;
    SceneGPU* d_world = nullptr;
    checkCudaErrors(cudaMallocManaged(&d_world, sizeof(SceneGPU)));

    Object** d_objects = nullptr;
    int numObjects = 0;

    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));

    // Initialize random states
    dim3 blocks_init((w + tx - 1) / tx, (h + ty - 1) / ty);
    dim3 threads_init(tx, ty);
    render_init << <blocks_init, threads_init >> > (w, h, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Load scene and setup objects and camera
    loadSceneToGPUFromFile("Scene.txt", d_world, &d_objects, &numObjects, &d_camera, w, h, d_rand_state);

    // Launch rendering kernel
    dim3 blocks((w + tx - 1) / tx, (h + ty - 1) / ty);
    dim3 threads(tx, ty);

    start = clock();
    render << <blocks, threads >> > (fb, w, h, ns, &d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

    std::cerr << "Rendering took " << timer_seconds << " seconds.\n";

    // Copy framebuffer to host memory (img)
    cudaMemcpy(img, fb, num_pixels * sizeof(Vec3), cudaMemcpyDeviceToHost);

    // Cleanup
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects)); // Note: objects inside d_objects are device allocated, memory leak unless handled inside SceneGPU destructor
    checkCudaErrors(cudaFree(d_rand_state));
    // d_camera allocated device memory, but no deletion kernel, possible leak
}