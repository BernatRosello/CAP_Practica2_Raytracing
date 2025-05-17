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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "raytracing.h"

#include "Vec3.h"
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


Scene randomScene()
{
	Scene list;
	list.add(new Object(
		new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f),
		new Diffuse(Vec3(0.5f, 0.5f, 0.5f))));

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			float choose_mat = random();
			Vec3 center(a + 0.9f * random(), 0.2f, b + 0.9f * random());
			if ((center - Vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
			{
				if (choose_mat < 0.8f)
				{ // diffuse
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Diffuse(Vec3(random() * random(),
										 random() * random(),
										 random() * random()))));
				}
				else if (choose_mat < 0.95f)
				{ // metallic
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Metallic(Vec3(0.5f * (1.0f + random()),
										  0.5f * (1.0f + random()),
										  0.5f * (1.0f + random())),
									 0.5f * random())));
				}
				else
				{ // crystalline
					list.add(new Object(
						new Sphere(center, 0.2f),
						new Crystalline(1.5f)));
				}
			}
		}
	}

	list.add(new Object(
		new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f),
		new Crystalline(1.5f)));
	list.add(new Object(
		new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f),
		new Diffuse(Vec3(0.4f, 0.2f, 0.1f))));
	list.add(new Object(
		new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f),
		new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)));

	return list;
}

void rayTracingCPU(Vec3 *img, int w, int h, int ns = 10, const std::string &filename = "")
{
	Scene world;

	if (filename.c_str() != "")
	{
		world = loadObjectsFromFile(filename);
	}
	else {
		world = randomScene();
	}
	world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
	world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

	Vec3 lookfrom(13.0f, 2.0f, 3.0f);
	Vec3 lookat(0.0f, 0.0f, 0.0f);
	float dist_to_focus = 10.0f;
	float aperture = 0.1f;

	Camera cam(lookfrom, lookat, Vec3(0.0f, 1.0f, 0.0f), 20.0f, float(w) / float(h), aperture, dist_to_focus);

	for (int j = h - 1; j >= 0; j--)
	{
		for (int i = 0; i < w; i++)
		{
			Vec3 col(0.0f, 0.0f, 0.0f);
			for (int s = 0; s < ns; s++)
			{
				float u = float(i + random()) / float(w);
				float v = float(j + random()) / float(h);
				Ray r = cam.get_ray(u, v);
				col += world.getSceneColor(r);
			}
			col /= float(ns);
			col[0] = sqrt(col[0]);
			col[1] = sqrt(col[1]);
			col[2] = sqrt(col[2]);
			img[j * w + i] = col;
		}
	}
}

int main()
{
	int w = 512; // 1200;
	int h = 256; // 800;
	int ns = 2;
	clock_t start, stop;
	double timer_seconds;

	size_t size = sizeof(unsigned char) * w * h * 3;
	unsigned char *data = (unsigned char *)malloc(size);

	Vec3 *img;
	size_t isize = w * h * sizeof(Vec3);
	cudaMallocManaged((void **)&img, isize);

	std::cerr << "--- CPU ---\n";
	start = clock();
	rayTracingCPU(img, w, h, ns, "Scene2.txt");

	for (int i = h - 1; i >= 0; i--)
	{
		for (int j = 0; j < w; j++)
		{
			size_t idx = i * w + j;
			data[idx * 3 + 0] = char(255.99 * img[idx].b());
			data[idx * 3 + 1] = char(255.99 * img[idx].g());
			data[idx * 3 + 2] = char(255.99 * img[idx].r());
		}
	}
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "CPU took " << timer_seconds << " seconds.\n\n";

	writeBMP("imgCPU-prueba.bmp", data, w, h);
	printf("Imagen CPU creada.\n");

	std::cerr << "--- GPU ---\n";
	start = clock();
	rayTracingGPU(img, w, h, ns, "Scene2.txt");

	for (int i = h - 1; i >= 0; i--)
	{
		for (int j = 0; j < w; j++)
		{
			size_t idx = i * w + j;
			data[idx * 3 + 0] = char(255.99 * img[idx].b());
			data[idx * 3 + 1] = char(255.99 * img[idx].g());
			data[idx * 3 + 2] = char(255.99 * img[idx].r());
		}
	}
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "GPU took " << timer_seconds << " seconds.\n";

	writeBMP("imgGPU-prueba.bmp", data, w, h);
	printf("Imagen GPU creada.\n");

	free(data);
	cudaDeviceReset();

	getchar();
	return (0);
}
