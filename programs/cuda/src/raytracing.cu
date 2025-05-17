// #include <float.h>
#include <cstdio>
// #include <cstdlib>
// #include <limits>

#include <ctime>
#include <fstream>
#include <sstream>

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
#include "raytracing.h"
// #include "utils.h"

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

__global__ void init_scene_list_and_cam(Object** aux, SceneGPU* d_world, int numobjects, Camera** d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world->setList(aux, numobjects);
        d_world->setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
        d_world->setInfColor(Vec3(1.0f, 1.0f, 1.0f));

        // CAMERA PLACEMENT
        Vec3 lookfrom(13.0f, 2.0f, 3.0f);
        Vec3 lookat(0.0f, 0.0f, 0.0f);
        float dist_to_focus = 10.0f;
        float aperture = 0.1f;

        *d_camera = new Camera(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx) / float(ny), aperture, dist_to_focus);
    }
}

__global__ void create_scene_from_data(SceneGPU* d_world, SphereData* sphere_data, int numobjects)
{
    //size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int idx = 0; idx < numobjects; idx++)
    {
        Object* o;
        switch (sphere_data[idx].material.type) {
        case MaterialType::DIFFUSE:
            o = new Object(
                new Sphere(
                    Vec3(sphere_data[idx].center[0],
                        sphere_data[idx].center[1],
                        sphere_data[idx].center[2]),
                    sphere_data[idx].radius
                ),
                new Diffuse(
                    Vec3(sphere_data[idx].material.color[0],
                        sphere_data[idx].material.color[1],
                        sphere_data[idx].material.color[2])
                )
            );
            break;
        case MaterialType::METALLIC:
            o = new Object(
                new Sphere(
                    Vec3(sphere_data[idx].center[0],
                        sphere_data[idx].center[1],
                        sphere_data[idx].center[2]),
                    sphere_data[idx].radius
                ),
                new Metallic(
                    Vec3(sphere_data[idx].material.color[0],
                        sphere_data[idx].material.color[1],
                        sphere_data[idx].material.color[2]),
                    sphere_data[idx].material.mat_property
                )
            );
            break;
        case MaterialType::CRYSTALLINE:
            o = new Object(
                new Sphere(
                    Vec3(sphere_data[idx].center[0],
                        sphere_data[idx].center[1],
                        sphere_data[idx].center[2]),
                    sphere_data[idx].radius
                ),
                new Crystalline(sphere_data[idx].material.mat_property)
            );
            break;
        }
        //d_world->addAt(idx, o);
        d_world->add(o);
    }

}

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
        *d_camera = new Camera(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx) / float(ny), aperture, dist_to_focus);
    }
}

void loadGPUSceneFromFile(const std::string& filename, int w, int h, SceneGPU *&d_world, Camera **&d_camera, Object **&aux, curandState *& d_rand_state) {
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo: " << filename << std::endl;
        return;
    }

    std::vector<SphereData> h_sphere_data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (ss >> token) {
            tokens.push_back(token);
        }

        if (tokens.empty()) continue; // Línea vacía

        // Esperamos al menos la palabra clave "Object"
        if (tokens[0] == "Object" && tokens.size() >= 12) { // Mínimo para Sphere y un material con 1 float
            // Parsear la esfera
            if (tokens[1] == "Sphere" && tokens[2] == "(" && tokens[7] == ")") {
                try {
                    float sx = std::stof(tokens[3].substr(tokens[3].find('(') + 1, tokens[3].find(',') - tokens[3].find('(') - 1));
                    float sy = std::stof(tokens[4].substr(0, tokens[4].find(',')));
                    float sz = std::stof(tokens[5].substr(0, tokens[5].find(',')));
                    float sr = std::stof(tokens[6]);

                    // Parsear el material del último objeto creado

                    if (tokens[8] == "Crystalline" && tokens[9] == "(" && tokens[11].back() == ')') {
                        float ma = std::stof(tokens[10]);
                        h_sphere_data.push_back(SphereData(sx, sy, sz, sr, MaterialType::CRYSTALLINE, .0f, .0f, .0f, ma));
                        std::cout << "Crystaline" << sx << " " << sy << " " << sz << " " << sr << " " << ma << "\n";
                    }
                    else if (tokens[8] == "Metallic" && tokens.size() == 15 && tokens[9] == "(" && tokens[14] == ")") {
                        float ma = std::stof(tokens[10].substr(tokens[10].find('(') + 1, tokens[10].find(',') - tokens[10].find('(') - 1));
                        float mb = std::stof(tokens[11].substr(0, tokens[11].find(',')));
                        float mc = std::stof(tokens[12].substr(0, tokens[12].find(',')));
                        float mf = std::stof(tokens[13].substr(0, tokens[13].length() - 1));
                        h_sphere_data.push_back(SphereData(sx, sy, sz, sr, MaterialType::METALLIC, ma, mb, mc, mf));
                        std::cout << "Metallic" << sx << " " << sy << " " << sz << " " << sr << " " << ma << " " << mb << " " << mc << " " << mf << "\n";
                    }
                    else if (tokens[8] == "Diffuse" && tokens.size() == 14 && tokens[9] == "(" && tokens[13].back() == ')') {
                        float ma = std::stof(tokens[10].substr(tokens[10].find('(') + 1, tokens[10].find(',') - tokens[10].find('(') - 1));
                        float mb = std::stof(tokens[11].substr(0, tokens[11].find(',')));
                        float mc = std::stof(tokens[12].substr(0, tokens[12].find(',')));
                        h_sphere_data.push_back(SphereData(sx, sy, sz, sr, MaterialType::DIFFUSE, ma, mb, mc, 0));
                        std::cout << "Diffuse" << sx << " " << sy << " " << sz << " " << sr << " " << ma << " " << mb << " " << mc << "\n";
                    }
                    else {
                        std::cerr << "Error: Material desconocido o formato incorrecto en la línea: " << line << std::endl;
                    }
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Conversión inválida en la línea: " << line << " - " << e.what() << std::endl;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Error: Valor fuera de rango en la línea: " << line << " - " << e.what() << std::endl;
                }
            }
            else {
                std::cerr << "Error: Formato de esfera incorrecto en la línea: " << line << std::endl;
            }
        }
        else {
            std::cerr << "Error: Formato de objeto incorrecto en la línea: " << line << std::endl;
        }
    }
    file.close();


    checkCudaErrors(cudaMalloc((void**)&d_rand_state, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init << <1, 1 >> > (d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Initialized GPU d_rand_state(2)" << std::endl;

    // make our world of hitables & the camera
    int numobjects = h_sphere_data.size();
    SphereData *d_sphere_data;
    checkCudaErrors(cudaMalloc(&d_sphere_data, numobjects * sizeof(SphereData)));
    checkCudaErrors(cudaMemcpy(d_sphere_data, h_sphere_data.data(), numobjects * sizeof(SphereData), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&aux, numobjects * sizeof(Object*)));
    // Malloc World GPU
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Scene)));
    // Malloc Camera GPU
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    std::cout << "Initializing Scene and Camera in hte GPU" << std::endl;
    init_scene_list_and_cam << <1, 1 >> > (aux, d_world, numobjects, d_camera, w, h);

    //int threads = std::min(numobjects, 512);
    //int blocks = numobjects / threads;
    int threads = 1;
    int blocks = 1;
    std::cout << "Creating GPU Scene from Scene file:" << filename << "\t using " << blocks << " blocks of " << threads << " threads." << std::endl;
    create_scene_from_data << <blocks, threads >> > (d_world, d_sphere_data, numobjects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Scene created successfully!" << std::endl;
}

void rayTracingGPU(Vec3* img, int w, int h, int ns, const std::string& filename)
{
    int tx = 8;
    int ty = 8;
    clock_t start, stop;
    double timer_seconds;

    std::cerr << "Rendering a " << w << "x" << h << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = w * h;
    size_t fb_size = num_pixels * sizeof(Vec3);

    start = clock();
    // allocate FB
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    SceneGPU* d_world;
    Camera** d_camera;
    Object** aux;
    curandState* d_rand_state2;

    if (filename.length() > 0)
    {
        loadGPUSceneFromFile(filename, w, h, d_world, d_camera, aux, d_rand_state2);
    }
    else {
        std::cout << "Starting random Scene generation on GPU" << std::endl;
        checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

        // we need that 2nd random state to be initialized for the world creation
        rand_init << <1, 1 >> > (d_rand_state2);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // make our world of hitables & the camera
        int numobjects = 22 * 22 + 1 + 3;
        checkCudaErrors(cudaMalloc((void**)&aux, numobjects * sizeof(Object*)));
        // Malloc World GPU
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Scene)));
        // Malloc Camera GPU
        checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
        create_world << <1, 1 >> > (aux, numobjects, d_world, d_camera, w, h, d_rand_state2);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Loading took " << timer_seconds << " seconds.\n";

    start = clock();
    // Render our buffer
    dim3 blocks(w / tx + 1, h / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(w, h, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, w, h, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    start = clock();
    for (int i = h - 1; i >= 0; i--)
    {
        for (int j = 0; j < w; j++)
        {
            size_t pixel_index = i * w + j;
            img[pixel_index] = fb[pixel_index];
        }
    }
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Copy took " << timer_seconds << " seconds.\n";

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(aux));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
}
