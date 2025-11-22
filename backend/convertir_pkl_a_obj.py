# backend/convertir_pkl_a_obj.py (CORREGIDO)
import smplx
import torch
import trimesh
import os

def convertir_modelo(base_model_folder, model_type, output_path):
    """
    Carga un modelo SMPL desde un archivo .pkl y lo exporta a .obj.
    """
    print(f"Cargando el modelo '{model_type}' desde: {base_model_folder}")

    try:
        # Cargamos el modelo especificando el tipo explícitamente
        model = smplx.create(
            base_model_folder,
            model_type=model_type, # <-- Cambio clave
            gender='neutral',
            use_pca=False,
            ext='pkl'
        )

        vertices = model.v_template
        faces = model.faces
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Asegurarse de que el directorio de salida exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        
        print(f"¡ÉXITO! Modelo convertido y guardado en: {output_path}")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == '__main__':
    # Ruta a la carpeta que contiene las subcarpetas 'smpl', 'smplx', etc.
    assets_folder = 'backend/assets'
    # Ruta de salida para el archivo .obj
    output_obj = os.path.join(assets_folder, 'smpl', 'SMPL_NEUTRAL.obj')

    # Llamamos a la función con el tipo de modelo 'smpl'
    convertir_modelo(
        base_model_folder=assets_folder,
        model_type='smpl', 
        output_path=output_obj
    )