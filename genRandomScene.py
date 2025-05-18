import random
import math
import argparse

# Esferas base (predeterminadas)
base_spheres = [
    'Object Sphere ( (0.0, -1000.0, 0.0), 1000.0 ) Diffuse ( (0.5, 0.5, 0.5) )',
    'Object Sphere ( (4.0, 1.0, 0.0), 1.0 ) Metallic ( (0.7, 0.6, 0.5), 0.0 )',
    'Object Sphere ( (-4.0, 1.0, 0.0), 1.0 ) Diffuse ( (0.4, 0.2, 0.1) )',
    'Object Sphere ( (0.0, 1.0, 0.0), 1.0 ) Crystalline ( 1.5 )'
]

materials = ["Diffuse", "Metallic", "Crystalline"]

def random_vec3(radius, min_val=-10.0, max_val=10.0):
    return (
        round(random.uniform(min_val, max_val), 2),
        round(random.uniform(0, radius), 2),  # y entre 0.2 y 1 para evitar que floten o atraviesen el plano
        round(random.uniform(min_val, max_val), 2)
    )

def distance(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))

def sphere_not_overlapping(center, radius, spheres):
    for (c, r) in spheres:
        if distance(center, c) < (radius + r):
            return False
    return True

def generate_random_sphere(existing, max_attempts=100):
    for _ in range(max_attempts):
        radius = round(random.uniform(0.2, 0.5), 2)
        center = random_vec3(radius)
        if sphere_not_overlapping(center, radius, existing):
            material_type = random.choice(materials)
            if material_type == "Diffuse":
                color = tuple(round(random.uniform(0.0, 1.0), 2) for _ in range(3))
                mat_str = f'Diffuse ( ({color[0]}, {color[1]}, {color[2]}) )'
            elif material_type == "Metallic":
                color = tuple(round(random.uniform(0.0, 1.0), 2) for _ in range(3))
                fuzz = round(random.uniform(0.0, 0.3), 2)
                mat_str = f'Metallic ( ({color[0]}, {color[1]}, {color[2]}), {fuzz} )'
            elif material_type == "Crystalline":
                ref_idx = round(random.uniform(1.3, 1.7), 2)
                mat_str = f'Crystalline ( {ref_idx} )'
            obj_str = f'Object Sphere ( ({center[0]}, {center[1]}, {center[2]}), {radius} ) {mat_str}'
            return obj_str, (center, radius)
    return None, None

def generate_scene(num_random_spheres):
    scene_lines = list(base_spheres)
    existing_spheres = []

    # Añadir esferas base al sistema de colisión
    existing_spheres.append(((0.0, -1000.0, 0.0), 1000.0))
    existing_spheres.append(((4.0, 1.0, 0.0), 1.0))
    existing_spheres.append(((-4.0, 1.0, 0.0), 1.0))
    existing_spheres.append(((0.0, 1.0, 0.0), 1.0))

    while len(scene_lines) < num_random_spheres + len(base_spheres):
        sphere_str, info = generate_random_sphere(existing_spheres)
        if sphere_str:
            scene_lines.append(sphere_str)
            existing_spheres.append(info)
        else:
            print("No se pudo colocar una esfera sin solape tras varios intentos.")

    with open("Scene.txt", "w") as f:
        for line in scene_lines:
            f.write(line + "\n")

# Ejecutar con x esferas aleatorias
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera una escena con esferas no solapadas.")
    parser.add_argument("cantidad", type=int, help="Cantidad de esferas aleatorias a generar")
    args = parser.parse_args()
    generate_scene(args.cantidad)