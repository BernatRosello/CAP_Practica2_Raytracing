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


#include "Camera.h"
#include "Object.h"
#include "Scene.h"
#include "Sphere.h"
#include "Diffuse.h"
#include "Metallic.h"
#include "Crystalline.h"

#include "random.h"
#include "utils.h"

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
							std::cout << "Crystaline" << sx << " " << sy << " " << sz << " " << sr << " " << ma << "\n";
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
							std::cout << "Metallic" << sx << " " << sy << " " << sz << " " << sr << " " << ma << " " << mb << " " << mc << " " << mf << "\n";
						}
						else if (tokens[8] == "Diffuse" && tokens.size() == 14 && tokens[9] == "(" && tokens[13].back() == ')') {
							float ma = std::stof(tokens[10].substr(tokens[10].find('(') + 1, tokens[10].find(',') - tokens[10].find('(') - 1));
							float mb = std::stof(tokens[11].substr(0, tokens[11].find(',')));
							float mc = std::stof(tokens[12].substr(0, tokens[12].find(',')));
							list.add(new Object(
								new Sphere(Vec3(sx, sy, sz), sr),
								new Diffuse(Vec3(ma, mb, mc))
							));
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

void rayTracingCPU(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1) {
	if (pw == -1) pw = w;
	if (ph == -1) ph = h;
	int patch_w = pw - px;
	// Scene world = randomScene();
	Scene world = loadObjectsFromFile("Scene1.txt");
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

int main() {
	//srand(time(0));

	int w = 256;// 1200;
	int h = 256;// 800;
	int ns = 10;

	int patch_x_size = w;
	int patch_y_size = h;
	int patch_x_idx = 1;
	int patch_y_idx = 1;

	int size = sizeof(unsigned char) * patch_x_size * patch_y_size * 3;
	unsigned char* data = (unsigned char*)calloc(size, 1);

	int patch_x_start = (patch_x_idx - 1) * patch_x_size;
	int patch_x_end = patch_x_idx * patch_x_size;
	int patch_y_start = (patch_y_idx - 1) * patch_y_size;
	int patch_y_end = patch_y_idx * patch_y_size;

	rayTracingCPU(data, w, h, ns, patch_x_start, patch_y_start, patch_x_end, patch_y_end);

	writeBMP("imgCPUImg1.bmp", data, patch_x_size, patch_y_size);
	printf("Imagen creada.\n");

	free(data);
	getchar();
	return (0);
}
