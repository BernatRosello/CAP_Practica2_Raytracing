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

void rayTracingCPU(Vec3 *img, int w, int h, int ns = 10)
{
	Scene world = randomScene();
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
	int ns = 10;
	clock_t start, stop;
	double timer_seconds;

	size_t size = sizeof(unsigned char) * w * h * 3;
	unsigned char *data = (unsigned char *)malloc(size);

	Vec3 *img;
	size_t isize = w * h * sizeof(Vec3);
	cudaMallocManaged((void **)&img, isize);

	std::cerr << "--- CPU ---\n";
	start = clock();
	rayTracingCPU(img, w, h, ns);

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
	rayTracingGPU(img, w, h, ns);

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
