#pragma once

#include <vector>

#include <curand_kernel.h>

#include "Vec3.h"
#include "Ray.h"
#include "Object.h"

class Scene {
public:
	Scene(int depth = 50) : ol(), sky(), inf(), d(depth) {}
	Scene(const Scene& list) = default;

	void add(Object* o) { ol.push_back(o); }
	void setSkyColor(Vec3 sky) { this->sky = sky; }
	void setInfColor(Vec3 inf) { this->inf = inf; }

	Vec3 getSceneColor(const Ray& r);

protected:
	Vec3 getSceneColor(const Ray& r, int depth);

private:
	std::vector<Object*> ol;
	Vec3 sky;
	Vec3 inf;
	int d;
};

class SceneGPU {
public:
	__device__ SceneGPU() : ol(nullptr), capacity(0), size(0), sky(), inf() {}
	__device__ SceneGPU(Object** list, size_t nobjects, size_t tmax, int depth = 50) : ol(list), capacity(nobjects), size(0), sky(), inf() {}

	__device__ void setList(Object** list, int numobjets) { ol = list; capacity = numobjets; }
	__device__ void add(Object* o) { ol[size] = o; size++; }
	__device__ void setSkyColor(Vec3 sky) { this->sky = sky; }
	__device__ void setInfColor(Vec3 inf) { this->inf = inf; }
	__device__ void addAt(size_t idx, Object* o) { ol[idx] = o; }
	__device__ void setSize(size_t new_size) { size = new_size; }

	__device__ Vec3 getSceneColor(const Ray& r, curandState* local_rand_state) {
		Ray tempr = r;
		Vec3 tempv = Vec3(1.0, 1.0, 1.0);
		for (int i = 0; i < 50; i++) {
			CollisionData cd;
			Object* aux = nullptr;
			float closest = FLT_MAX;  // std::numeric_limits<float>::max();  // initially tmax = std::numeric_limits<float>::max()
			for (int j = 0; j < size; j++) {
				if (ol[j]->checkCollision(tempr, 0.001f, closest, cd)) { // tmin = 0.001
					aux = ol[j];
					closest = cd.time;
				}
			}

			if (aux) {
				Ray scattered;
				Vec3 attenuation;
				if (aux->scatter(tempr, cd, attenuation, scattered, local_rand_state)) {
					tempv *= attenuation;
					tempr = scattered;
				} else {
					return Vec3(0.0f, 0.0f, 0.0f);
				}
			}
			else {
				Vec3 unit_direction = unit_vector(tempr.direction());
				float t = 0.5f * (unit_direction.y() + 1.0f);
				Vec3 c = (1.0f - t) * inf + t * sky;
				return tempv * c;
			}
		}
		return Vec3(0.0, 0.0, 0.0);
	}


private:
	Object** ol;
	size_t capacity;
	size_t size;
	Vec3 sky;
	Vec3 inf;
};

enum MaterialType { DIFFUSE = 0, METALLIC = 1, CRYSTALLINE = 2 };

struct MaterialData {
	MaterialType type;
	float color[3];		// Used by DIFFUSE and METALLIC
	float mat_property; // Fuzz for METALLIC, IOR for CRYSTALLINE, unused for DIFFUSE

	MaterialData(MaterialType type, float r, float g, float b, float mat_property) :
		type(type), color{ r,g,b }, mat_property(mat_property) {}
};

struct SphereData {
	float center[3];
	float radius;
	MaterialData material;

	SphereData(float sx, float sy, float sz, float sr, MaterialType mat, float r, float g, float b, float mat_property) :
		center{ sx,sy,sz }, radius(sr), material(mat, r, g, b, mat_property) {}
};
