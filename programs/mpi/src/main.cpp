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

#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <sstream>
#include <fstream>
#include <ios>

#include "Camera.h"
#include "Object.h"
#include "Scene.h"
#include "Sphere.h"
#include "Diffuse.h"
#include "Metallic.h"
#include "Crystalline.h"

#include "random.h"
#include "utils.h"
#include <chrono>
#include <mpi.h>

//#define LOG 0
#define ROW 1
#define COL 2
#define RECT 3

Scene loadObjectsFromFile(const std::string& filename) {
	std::ifstream file(filename);
	std::string line;

	Scene list;

	if (file.is_open()) {
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
							list.add(new Object(
								new Sphere(Vec3(sx, sy, sz), sr),
								new Crystalline(ma)
							));
							//std::cout << "Crystaline" << sx << " " << sy << " " << sz << " " << sr << " " << ma << "\n";
						}
						else if (tokens[8] == "Metallic" && tokens.size() == 15 && tokens[9] == "(" && tokens[14] == ")") {
							float ma = std::stof(tokens[10].substr(tokens[10].find('(') + 1, tokens[10].find(',') - tokens[10].find('(') - 1));
							float mb = std::stof(tokens[11].substr(0, tokens[11].find(',')));
							float mc = std::stof(tokens[12].substr(0, tokens[12].find(',')));
							float mf = std::stof(tokens[13].substr(0, tokens[13].length() - 1));
							list.add(new Object(
								new Sphere(Vec3(sx, sy, sz), sr),
								new Metallic(Vec3(ma, mb, mc), mf)
							));
							//std::cout << "Metallic" << sx << " " << sy << " " << sz << " " << sr << " " << ma << " " << mb << " " << mc << " " << mf << "\n";
						}
						else if (tokens[8] == "Diffuse" && tokens.size() == 14 && tokens[9] == "(" && tokens[13].back() == ')') {
							float ma = std::stof(tokens[10].substr(tokens[10].find('(') + 1, tokens[10].find(',') - tokens[10].find('(') - 1));
							float mb = std::stof(tokens[11].substr(0, tokens[11].find(',')));
							float mc = std::stof(tokens[12].substr(0, tokens[12].find(',')));
							list.add(new Object(
								new Sphere(Vec3(sx, sy, sz), sr),
								new Diffuse(Vec3(ma, mb, mc))
							));
							//std::cout << "Diffuse" << sx << " " << sy << " " << sz << " " << sr << " " << ma << " " << mb << " " << mc << "\n";
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
	}
	else {
		std::cerr << "Error: No se pudo abrir el archivo: " << filename << std::endl;
	}
	return list;
}


Scene randomScene() {
	int n = 500;
	Scene list;
	list.add(new Object(
		new Sphere(Vec3(0, -1000, 0), 1000),
		new Diffuse(Vec3(0.5, 0.5, 0.5))
	));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = Mirandom();
			Vec3 center(a + 0.9f * Mirandom(), 0.2f, b + 0.9f * Mirandom());
			if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
				if (choose_mat < 0.8f) {  // diffuse
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Diffuse(Vec3(Mirandom() * Mirandom(),
							Mirandom() * Mirandom(),
							Mirandom() * Mirandom()))
					));
				}
				else if (choose_mat < 0.95f) { // metal
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Metallic(Vec3(0.5f * (1 + Mirandom()),
							0.5f * (1 + Mirandom()),
							0.5f * (1 + Mirandom())),
							0.5f * Mirandom())
					));
				}
				else {  // glass
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Crystalline(1.5f)
					));
				}
			}
		}
	}

	list.add(new Object(
		new Sphere(Vec3(0, 1, 0), 1.0),
		new Crystalline(1.5f)
	));
	list.add(new Object(
		new Sphere(Vec3(-4, 1, 0), 1.0f),
		new Diffuse(Vec3(0.4f, 0.2f, 0.1f))
	));
	list.add(new Object(
		new Sphere(Vec3(4, 1, 0), 1.0f),
		new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)
	));

	return list;
}

void rayTracingCPU(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1, const std::string& filename = "") {
	if (pw == -1) pw = w;
	if (ph == -1) ph = h;
	int patch_w = pw - px;

	Scene world;

	if (!filename.empty())
	{
		world = loadObjectsFromFile(filename);
	}
	else {
		world = randomScene();
	}

	world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
	world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

	Vec3 lookfrom(13, 2, 3);
	Vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.1f;

	Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);
	
	for (int j = 0; j < (ph - py); j++) {
		for (int i = 0; i < (pw - px); i++) {

			Vec3 col(0, 0, 0);
			for (int s = 0; s < ns; s++) {
				float u = float(i + px + Mirandom()) / float(w);
				float v = float(j + py + Mirandom()) / float(h);
				Ray r = cam.get_ray(u, v);
				col += world.getSceneColor(r);
			}
			col /= float(ns);
			col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

			img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
			img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
			img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
		}
	}
}

int main(int argc, char** argv) {

	int render_type = 0;
	std::string render_arg = argv[1];
	if (render_arg == "r") render_type = ROW;
	else if (render_arg == "c") render_type = COL;
	else {
		std::cout << "Render type desconocido. Usa 'r' para rows o 'c' para columns.\n";
		return EXIT_FAILURE;
	}

	//srand(time(0));
	int pid, np;
	const int root = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	int w = 256;// 1200;
	int h = 256;// 800;
	int ns = 10;

	/*
	if (pid == root) {
		for (int i = 0; i < argc; i++) {
			std::cout << "Argv[" << i << "] " << argv[i] << std::endl;
		}
	}
	*/
	
	std::string filename;
	int num_spheres = 0;
	// Parseo de argumentos opcionales
	for (int i = 5; i < argc; ++i) {
		std::string arg = argv[i];
		if (i == 5) {
			num_spheres = std::stoi(arg);
		}
		else if (i == 6) {
			if (std::stoi(arg) % 8 != 0) {
				std::cout << "Width must be multiple of 8" << std::endl;
				return EXIT_FAILURE;
			}
			w = std::stoi(arg);
		}
		else if (i == 7) {
			h = std::stoi(arg);
		}
		else if (i == 8) {
			ns = std::stoi(arg);
		}
		else if (i == 9) {
			filename = arg;
		}
	}

	int patch_x_size, patch_y_size, patch_x_idx, patch_y_idx;
	
	if (render_type == ROW) {
		if (pid == root) {
			std::cout << np << " processors" << std::endl;
			std::cout << "Render process by rows." << std::endl;
		}

		patch_x_size = w;
		patch_x_idx = 0;
		patch_y_size = h / np;
		patch_y_idx = pid;
	}
	else if (render_type == COL) {
		if (pid == root) {
			std::cout << np << " processors" << std::endl;
			std::cout << "Render process by columns." << std::endl;
		}

		patch_x_size = w / np;
		patch_x_idx = pid;
		patch_y_size = h;
		patch_y_idx = 0;
	}

	int size = sizeof(unsigned char) * patch_x_size * patch_y_size * 3;
	unsigned char* data = (unsigned char*)calloc(size, 1);

	int patch_x_start = patch_x_idx * patch_x_size;
	int patch_x_end = (patch_x_idx+1) * patch_x_size;
	int patch_y_start = patch_y_idx * patch_y_size;
	int patch_y_end = (patch_y_idx + 1) * patch_y_size;

	std::chrono::duration<double> elapsed;
	auto start = std::chrono::high_resolution_clock::now();

	if (!filename.empty()) {
		if (pid == root) {
			std::cout << "Scene from filename used." << std::endl;
		}
		rayTracingCPU(data, w, h, ns, patch_x_start, patch_y_start, patch_x_end, patch_y_end, filename);
	}
	else {
		rayTracingCPU(data, w, h, ns, patch_x_start, patch_y_start, patch_x_end, patch_y_end);
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	elapsed = (end - start);

	int full_size = size * np;
	unsigned char* full_data = (unsigned char*)calloc(full_size, 1);

	MPI_Gather(data, size, MPI_CHAR, full_data, size, MPI_CHAR, root, MPI_COMM_WORLD);

	double max_t, min_t, sum_t;

	MPI_Reduce(&elapsed, &min_t, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed, &sum_t, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	if (pid == root) {
		double avg_t = sum_t / np;
		char* wr_render_type;
		if (render_type == ROW) {
			wr_render_type = (char*)malloc(sizeof(char) * 3);
			wr_render_type = "row";
		}
		else if (render_type == COL) {
			wr_render_type = (char*)malloc(sizeof(char) * 3);
			wr_render_type = "col";
		}
		writeCSV("results_mpi.csv", np, w, h, num_spheres, ns, wr_render_type, min_t, max_t, avg_t);

		unsigned char* image_data;

		if (render_type == ROW) {
			image_data = full_data;
		}
		if (render_type == COL) {
			image_data = (unsigned char*)calloc(full_size, 1);

			for (int i = 0; i < np; ++i) {
				int col_offset = i * patch_x_size;
				for (int y = 0; y < h; ++y) {
					for (int x = 0; x < patch_x_size; ++x) {
						int src_idx = (i * patch_x_size * h + y * patch_x_size + x) * 3;
						int dst_idx = (y * w + col_offset + x) * 3;
						image_data[dst_idx + 0] = full_data[src_idx + 0];
						image_data[dst_idx + 1] = full_data[src_idx + 1];
						image_data[dst_idx + 2] = full_data[src_idx + 2];
					}
				}
			}
		}

		writeBMP("imgCPU_MPI.bmp", image_data, w, h);
		
		printf("Imagen creada.\n");
	}

#ifdef LOG
	std::cout << "[" << pid << "] patch_x_size = " << patch_x_size << std::endl;
	std::cout << "[" << pid << "] patch_y_size = " << patch_y_size << std::endl;
	std::cout << "[" << pid << "] size = " << size << std::endl;
	std::cout << "[" << pid << "] patch_x_start = " << patch_x_start << std::endl;
	std::cout << "[" << pid << "] patch_x_end = " << patch_x_end << std::endl;
	std::cout << "[" << pid << "] patch_y_start = " << patch_y_start << std::endl;
	std::cout << "[" << pid << "] patch_y_end = " << patch_y_end << std::endl;
	std::cout << "[" << pid << "] Elapsed Time = " << elapsed.count() << std::endl;
#endif

	free(full_data);
	free(data);
	//getchar();
	MPI_Finalize();
	return (0);
}
