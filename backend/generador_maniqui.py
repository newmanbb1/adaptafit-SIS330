# backend/generador_maniqui.py
import trimesh
import numpy as np
import os

def crear_maniqui_base(ruta_salida="mannequin_base.obj"):
    """
    Crea un maniquí 3D simple a partir de primitivas y lo guarda como un archivo .obj.
    """
    print("Creando maniquí 3D base...")

    # Definir partes del cuerpo como esferas y cilindros
    # Cabeza (esfera)
    cabeza = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    cabeza.apply_translation([0, 1.7, 0])

    # Torso (cilindro)
    torso = trimesh.creation.cylinder(radius=0.15, height=0.5, sections=32)
    torso.apply_translation([0, 1.35, 0])

    # Cadera (esfera)
    cadera = trimesh.creation.icosphere(subdivisions=3, radius=0.16)
    cadera.apply_translation([0, 1.1, 0])

    # Piernas y brazos (cilindros)
    # Esto es una simplificación. Un modelo real conectaría las articulaciones.
    pierna_izq = trimesh.creation.cylinder(radius=0.06, height=1.0, sections=32)
    pierna_izq.apply_translation([-0.1, 0.6, 0])

    pierna_der = trimesh.creation.cylinder(radius=0.06, height=1.0, sections=32)
    pierna_der.apply_translation([0.1, 0.6, 0])

    brazo_izq = trimesh.creation.cylinder(radius=0.05, height=0.6, sections=32)
    brazo_izq.apply_translation([-0.25, 1.4, 0])
    brazo_izq.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(90), [0, 0, 1]))

    brazo_der = trimesh.creation.cylinder(radius=0.05, height=0.6, sections=32)
    brazo_der.apply_translation([0.25, 1.4, 0])
    brazo_der.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0, 0, 1]))

    # Unir todas las partes en una sola malla
    maniqui_completo = trimesh.util.concatenate([
        cabeza, torso, cadera, pierna_izq, pierna_der, brazo_izq, brazo_der
    ])

    # Exportar a un archivo .obj
    maniqui_completo.export(ruta_salida)

    print(f"Maniquí guardado exitosamente en: {ruta_salida}")

if __name__ == '__main__':
    # Guardamos el maniquí en una carpeta de assets para tenerlo ordenado
    output_dir = "assets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    crear_maniqui_base(ruta_salida=os.path.join(output_dir, "mannequin_base.obj"))