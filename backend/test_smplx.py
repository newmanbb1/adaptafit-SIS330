import smplx
import torch
import os
import traceback

def final_test():
    print("--- Iniciando Prueba de Carga Final ---")
    try:
        # Definimos la ruta al directorio de modelos, relativa al script
        # __file__ es la ruta de este script (test_smplx.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'assets', 'smpl')
        
        print(f"Buscando 'SMPL_NEUTRAL.pkl' en el directorio: {model_dir}")
        
        if not os.path.exists(os.path.join(model_dir, 'SMPL_NEUTRAL.pkl')):
            print("\n¡ERROR CRÍTICO! No se encontró el archivo 'SMPL_NEUTRAL.pkl' en la carpeta.")
            print("Asegúrate de haber descargado la versión 1.0.0 y renombrado el archivo correctamente.")
            return

        # El comando de carga exacto que usa la aplicación
        model = smplx.create(model_path=model_dir, 
                             model_type='smpl',
                             gender='neutral')
        
        print("\n******************************************")
        print("¡¡¡ÉXITO!!! El modelo se ha cargado.")
        print(f"Tipo de objeto: {type(model)}")
        print("******************************************")

    except Exception as e:
        print("\n--- ¡ERROR DEFINITIVO DETECTADO! ---")
        print("Este es el traceback completo que necesitamos:")
        traceback.print_exc()

if __name__ == "__main__":
    final_test()