#pragma once

#include "Vec3.h"

void writeBMP(const char* filename, unsigned char* data, int w, int h);
void writeCSV(const char* filename, int np, int th, int w, int h, int num_spheres, int ns,  char* render_type, double min_t, double max_t, double avg_t);

float schlick(float cosine, float ref_idx);
bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted);
Vec3 reflect(const Vec3& v, const Vec3& n);
