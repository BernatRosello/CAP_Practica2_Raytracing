#pragma once

#include "Vec3.h"

void rayTracingGPU(Vec3* img, int w, int h, int ns = 10, const std::string& filename = "");
