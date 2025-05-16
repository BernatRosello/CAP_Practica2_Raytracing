#pragma once

enum MaterialType { DIFFUSE = 0, METALLIC = 1, CRYSTALLINE = 2 };

struct MaterialData {
    MaterialType type;
    float color[3];  // Used by DIFFUSE and METALLIC
    float mat_property;      // Fuzz for METALLIC, IOR for CRYSTALLINE, unused for DIFFUSE
};

struct SphereData {
    float center[3];
    float radius;
    MaterialData material;
};