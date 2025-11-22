# --- 1. Importaciones ---
import os
import shutil
import json
from datetime import date,timedelta
from typing import Optional,List
import cv2
import mediapipe as mp
import numpy as np
import trimesh
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, BackgroundTasks,status
from fastapi.security import OAuth2PasswordRequestForm,OAuth2PasswordBearer
from sqlalchemy.orm import Session
from . import models, database, schemas,auth_utils
import pickle
import traceback
from scipy.spatial.transform import Rotation as R
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.staticfiles import StaticFiles 
import time
try:
    import open3d as o3d
except ImportError:
    print("Dependencia no encontrada: Por favor, instala open3d con 'pip install open3d'")
import smplx
import re
import base64
import requests
from urllib.parse import quote
from fastapi.responses import StreamingResponse

def clean_numpy_types(data):
    """
    Recorre un diccionario o lista y convierte recursivamente 
    los tipos de numpy a tipos nativos de Python.
    """
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {key: clean_numpy_types(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [clean_numpy_types(item) for item in data]
    return data


# --- 2. Configuración Inicial ---
# Dejamos el Plan B (clave directa) por ahora para confirmar que funciona
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

RAPIDAPI_KEY = ""
if not RAPIDAPI_KEY:
    print("ADVERTENCIA: RAPIDAPI_KEY no está configurada en el entorno.")
    # Puedes poner una clave de fallback si quieres, pero no es recomendado
    # RAPIDAPI_KEY = "tu-clave-aqui"

# --- Añade esto ---
MESHY_API_KEY =  "" #
if not MESHY_API_KEY :
    print("ADVERTENCIA: MESHY_API_KEY no está configurada. Se omitirá la generación de Meshy 3D.")
# --- Fin de añadido ---  


# Creación de la App y Tablas de BD (solo una vez)
database.Base.metadata.create_all(bind=database.engine)
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

origins = [
    "http://localhost:5173",  # El puerto por defecto de Vite/React
    "http://localhost:3000",  # El puerto por defecto de Create React App
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Permite estos orígenes
    allow_credentials=True,
    allow_methods=["*"],         # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],         # Permite todos los headers
)
# Esto permite que el frontend acceda a los archivos en la carpeta 'jobs'
app.mount("/jobs", StaticFiles(directory="jobs"), name="jobs")

# Instancias de MediaPipe (solo una vez)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- 3. Definiciones de Constantes y Mapas ---
# --- REEMPLAZA TU MAPA (v2) POR ESTE (v3) ---
BODY_PART_JOINT_MAP = {
    'torso_completo': [0, 3, 6, 9, 12, 13, 14, 15], # Pelvis, Columna, Clavículas, Cuello, Cabeza
    'pierna_izquierda': [1, 4, 7, 10],            # Cadera Izq, Rodilla, Tobillo, Pie
    'pierna_derecha': [2, 5, 8, 11],             # Cadera Der, Rodilla, Tobillo, Pie
    'brazo_izquierdo': [16, 18, 20, 22],          # Hombro Izq, Codo, Muñeca, Mano
    'brazo_derecho': [17, 19, 21, 23]           # Hombro Der, Codo, Muñeca, Mano
}
MP_JOINT_MAP = {
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28
}
BONES_TO_PROCESS = [
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")
]

# --- 4. Funciones Auxiliares y de Lógica Principal ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()
def get_user_by_email(db: Session, email: str):
    """Busca un usuario en la BD por su email."""
    return db.query(models.User).filter(models.User.email == email).first()

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
) -> models.User:
    """
    Dependencia que decodifica el token, valida al usuario
    y devuelve el objeto User de la BD.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. Decodificar el token
    payload = auth_utils.decode_token(token)
    if payload is None:
        raise credentials_exception

    # 2. Obtener el email del "subject" del token
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception

    # 3. Obtener el usuario de la BD
    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception

    # 4. Devolver el objeto User
    return user


def draw_landmarks_on_image(image_path, output_path):
    # ... (tu función sin cambios)
    image = cv2.imread(image_path)
    if image is None:
        return False
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            cv2.imwrite(output_path, image)
            return True
    return False

# --- 4. Funciones Auxiliares y de Lógica Principal ---
# ... (tus otras funciones auxiliares) ...

# (Pega esto donde estaba la función anterior)

def generar_modelo_3d_con_meshy(image_path: str, job_dir: str, job_id: str) -> Optional[str]:
    """
    Llama a la API de Meshy (OpenAPI v1 - Image-to-3D), espera el resultado y
    descarga el modelo .glb en el directorio del job.
    
    Devuelve la URL relativa para el frontend.
    """
    if not MESHY_API_KEY:
        print("Saltando Meshy: API Key no configurada.")
        return None
        
    print(f"🤖 Iniciando generación de modelo 3D con Meshy API (OpenAPI v1) para: {job_id}")
    
    # --- 1. Leer y codificar la imagen a Base64 Data URI ---
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # Asumimos que es jpeg, como se guarda en el pipeline
        data_uri = f"data:image/jpeg;base64,{encoded_string}" 
    except Exception as e:
        print(f"❌ ERROR: No se pudo leer o codificar la imagen en Base64: {e}")
        return None

    # --- 2. Iniciar la Tarea (POST con JSON) ---
    # ¡¡NUEVO ENDPOINT de la documentación!!
    url_post = "https://api.meshy.ai/openapi/v1/image-to-3d" 
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
        "Content-Type": "application/json" # ¡¡MUY IMPORTANTE!!
    }
    
    # ¡¡NUEVO PAYLOAD (JSON)!!
    # Usamos los parámetros que recomienda la doc (enable_pbr, should_texture)
    payload = {
        "image_url": data_uri,  # Usamos el Data URI
        "enable_pbr": True,     # Pedimos mapas PBR
        "should_texture": True, # Nos aseguramos de que genere texturas
        "should_remesh": True   # Asegura que la topología sea buena
    }

    try:
        # Usamos 'json=payload' para enviar como application/json
        response_post = requests.post(url_post, headers=headers, json=payload) 
        response_post.raise_for_status()
        
        # La nueva doc dice que la respuesta es {"result": "task_id_string"}
        task_id = response_post.json().get('result') 
        if not task_id:
            print(f"❌ ERROR Meshy: No se recibió 'result' (task_id). Respuesta: {response_post.text}")
            return None
        print(f"🤖 Tarea de Meshy creada. Task ID: {task_id}")

    except requests.RequestException as e:
        print(f"❌ ERROR Meshy (POST): {e}")
        if e.response: print(f"Respuesta: {e.response.text}")
        return None
    except Exception as e_json:
        print(f"❌ ERROR Meshy (JSON Parse): {e_json}. Respuesta: {response_post.text}")
        return None


    # --- 3. Consultar el Estado (Polling GET) ---
    # ¡¡NUEVO ENDPOINT GET de la documentación!!
    url_get = f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}" 
    headers_get = {"Authorization": f"Bearer {MESHY_API_KEY}"}
    max_wait_time = 300 # Máximo 5 minutos de espera
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response_get = requests.get(url_get, headers=headers_get)
            response_get.raise_for_status()
            data = response_get.json()
            
            status = data.get('status')
            progress = data.get('progress', 0)
            print(f"🤖 Estado Meshy [Task: {task_id}]: {status} ({progress}%)")

            if status == 'SUCCEEDED':
                # --- 4. Descargar el Modelo ---
                model_url = data.get('model_urls', {}).get('glb')
                if not model_url:
                    print(f"❌ ERROR Meshy: Tarea exitosa pero no se encontró 'model_urls.glb'.")
                    return None
                
                print(f"🤖 ¡Éxito! Descargando modelo desde: {model_url}")
                response_download = requests.get(model_url, stream=True)
                response_download.raise_for_status()
                
                output_filename = "meshy_model.glb"
                output_path = os.path.join(job_dir, output_filename)
                
                with open(output_path, 'wb') as f:
                    for chunk in response_download.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Devuelve la URL relativa que el frontend puede usar (gracias a StaticFiles)
                relative_url = f"/jobs/{job_id}/{output_filename}"
                print(f"✅ Modelo de Meshy guardado en: {output_path}")
                print(f"✅ URL relativa para frontend: {relative_url}")
                return relative_url

            elif status == 'FAILED':
                error_msg = data.get('task_error', {}).get('message', 'Error desconocido')
                print(f"❌ ERROR Meshy: La tarea falló. Razón: {error_msg}")
                return None
            
            # Si está en 'PENDING' o 'IN_PROGRESS', esperar 10 segundos
            time.sleep(10)

        except requests.RequestException as e:
            print(f"❌ ERROR Meshy (GET Polling): {e}")
            if e.response: print(f"Respuesta: {e.response.text}")
            time.sleep(10) # Espera antes de reintentar
    
    print(f"❌ ERROR Meshy: Tiempo de espera agotado ({max_wait_time}s) para la tarea {task_id}.")
    return None


# --- REEMPLAZA TU FUNCIÓN (v2) POR ESTA (v3) ---
def generar_reporte_analisis_3d(obj_path: str, smpl_model_dir: str) -> dict:
    print("🧬 Iniciando análisis de métricas 3D del modelo (v3 Corregido)...")
    try:
        # --- 1. Carga y centrado del modelo ---
        mesh = trimesh.load(obj_path, force='mesh')
        centroid_xz = mesh.vertices[:, [0, 2]].mean(axis=0)
        mesh.vertices[:, 0] -= centroid_xz[0]
        mesh.vertices[:, 2] -= centroid_xz[1]
        
        if not mesh.is_watertight:
            print("ADVERTENCIA: La malla no es hermética. Rellenando agujeros...")
            mesh.fill_holes()
        mesh.fix_normals()
        
        verts = mesh.vertices
        ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
        altura_m = ymax - ymin
        volumen_total_m3 = abs(mesh.volume)

        # --- 2. Función de cálculo de circunferencia (sin cambios) ---
        # --- REEMPLAZA TU FUNCIÓN 'calcular_circunferencia' POR ESTA (v4) ---

        def calcular_circunferencia(mesh, 
                                    altura_proporcional: float, 
                                    ymin_global: float = None, 
                                    altura_global: float = None) -> float:
            """
            Calcula la circunferencia de un mesh.
            - Si se provee ymin_global y altura_global, usa el método de "altura global".
            - Si no, usa el método de "altura local" (basado en la propia altura del mesh).
            """
            if mesh is None:
                # print(f"ADVERTENCIA: Intento de medir circunferencia en una parte del cuerpo faltante.")
                return 0.0
            
            try:
                # Método 1: Corte basado en la altura GLOBAL (para el Torso)
                if ymin_global is not None and altura_global is not None:
                    y_corte = ymin_global + altura_proporcional * altura_global
                
                # Método 2: Corte basado en la altura LOCAL (para Brazos y Piernas)
                else:
                    verts = mesh.vertices
                    ymin_local = verts[:, 1].min()
                    ymax_local = verts[:, 1].max()
                    altura_local = ymax_local - ymin_local
                    if altura_local < 1e-6: return 0.0 # Evitar división por cero
                    y_corte = ymin_local + altura_proporcional * altura_local

                # Realiza el corte
                line_segments = trimesh.intersections.mesh_plane(
                    mesh=mesh,
                    plane_normal=[0, 1, 0],
                    plane_origin=[0, y_corte, 0]
                )
                
                if not len(line_segments):
                    # print(f"Debug: No se encontró intersección en {altura_proporcional:.2f}")
                    return 0.0
                return np.sum([np.linalg.norm(p2 - p1) for p1, p2 in line_segments])

            except Exception as e:
                print(f"ADVERTENCIA: Falló el cálculo de circunferencia: {e}")
                return 0.0

        # --- 3. Segmentación del modelo (Usando el mapa v3) ---
        model = smplx.create(model_path=smpl_model_dir, model_type='smpl', gender='neutral')
        weights = model.lbs_weights.detach().cpu().numpy()
        
        part_influences = []
        part_keys = list(BODY_PART_JOINT_MAP.keys())
        for part_name in part_keys:
            joint_indices = BODY_PART_JOINT_MAP[part_name]
            influence = weights[:, joint_indices].sum(axis=1)
            part_influences.append(influence)
        
        vertex_labels = np.argmax(np.stack(part_influences), axis=0)
        
        part_meshes = {}
        volumenes_partes_m3 = {}
        print("Segmentando el modelo en partes (v3)...")
        for i, part_name in enumerate(part_keys):
            face_mask = np.all(vertex_labels[mesh.faces] == i, axis=1)
            if np.any(face_mask):
                part_mesh = mesh.submesh([face_mask])[0]
                if not part_mesh.is_watertight:
                    part_mesh.fill_holes()
                part_meshes[part_name] = part_mesh
                volumenes_partes_m3[part_name] = abs(part_mesh.volume)
            else:
                part_meshes[part_name] = None
                volumenes_partes_m3[part_name] = 0.0
        
        print("Segmentación completada.")

        # --- 4. Cálculo de Métricas (Lógica Corregida v3) ---
        
        # A. Circunferencias (usando los sub-modelos correctos)
        # (Los porcentajes de altura son relativos al suelo (ymin))
        
        # Torso
        # A. Circunferencias
        
        # Torso: Usa el método GLOBAL (pasando ymin y altura_m)
        circ_pecho_m = calcular_circunferencia(part_meshes.get('torso_completo'), 0.75, ymin, altura_m)
        circ_cintura_m = calcular_circunferencia(part_meshes.get('torso_completo'), 0.65, ymin, altura_m)
        circ_cadera_m = calcular_circunferencia(part_meshes.get('torso_completo'), 0.52, ymin, altura_m)

        # Piernas (Muslos): Usa el método LOCAL (al 50% de su *propia* altura)
        # (El muslo está aprox. a la mitad de la pierna (cadera-tobillo))
        circ_muslo_der_m = calcular_circunferencia(part_meshes.get('pierna_derecha'), 0.75)
        circ_muslo_izq_m = calcular_circunferencia(part_meshes.get('pierna_izquierda'), 0.75)

        # Brazos (Bíceps): Usa el método LOCAL (al 50% de su *propia* altura)
        # (El bíceps está aprox. a la mitad del brazo (hombro-muñeca))
        circ_brazo_der_m = calcular_circunferencia(part_meshes.get('brazo_derecho'), 0.25)
        circ_brazo_izq_m = calcular_circunferencia(part_meshes.get('brazo_izquierdo'), 0.25)
        # B. Índices Corporales
        indice_cintura_cadera = round(circ_cintura_m / circ_cadera_m, 2) if circ_cadera_m > 0 else 0
        indice_cintura_altura = round(circ_cintura_m / altura_m, 2) if altura_m > 0 else 0
        
        # C. Análisis de Simetría
        def calcular_desbalance(v_izq, v_der):
            if v_izq + v_der < 1e-6: return 0.0 # Evitar división por cero
            diff = v_izq - v_der
            avg = (v_izq + v_der) / 2
            return round((diff / avg) * 100, 2)

        vol_brazo_izq = volumenes_partes_m3.get('brazo_izquierdo', 0)
        vol_brazo_der = volumenes_partes_m3.get('brazo_derecho', 0)
        vol_pierna_izq = volumenes_partes_m3.get('pierna_izquierda', 0)
        vol_pierna_der = volumenes_partes_m3.get('pierna_derecha', 0)

        # --- 5. Reporte Final (v3) ---
        reporte = {
            "altura_cm": round(altura_m * 100, 2),
            "volumen_total_litros": round(volumen_total_m3 * 1000, 2),
            
            "circunferencias_cm": {
                "pecho": round(circ_pecho_m * 100, 2),
                "cintura": round(circ_cintura_m * 100, 2),
                "cadera": round(circ_cadera_m * 100, 2),
                "muslo_derecho": round(circ_muslo_der_m * 100, 2),
                "muslo_izquierdo": round(circ_muslo_izq_m * 100, 2),
                "brazo_derecho": round(circ_brazo_der_m * 100, 2),
                "brazo_izquierdo": round(circ_brazo_izq_m * 100, 2),
            },
            
            "volumenes_por_parte_litros": {
                k: round(v * 1000, 2) for k, v in volumenes_partes_m3.items()
            },
            
            "indices_corporales": {
                "indice_cintura_cadera": indice_cintura_cadera,
                "indice_cintura_altura": indice_cintura_altura,
            },
            
            "analisis_de_simetria_porc": {
                "desbalance_volumen_brazos": calcular_desbalance(vol_brazo_izq, vol_brazo_der),
                "desbalance_volumen_piernas": calcular_desbalance(vol_pierna_izq, vol_pierna_der),
                "desbalance_circunf_brazos": calcular_desbalance(circ_brazo_izq_m, circ_brazo_der_m),
                "desbalance_circunf_piernas": calcular_desbalance(circ_muslo_izq_m, circ_muslo_der_m),
            }
        }
        
        print("✅ Reporte de análisis 3D (v3) generado exitosamente.")
        return reporte

    except Exception as e:
        print(f"❌ ERROR GRAVE durante la generación del reporte 3D (v3): {e}")
        traceback.print_exc()
        return None

def analizar_y_deformar_modelo_smplx(job_dir: str, landmarks_path: str, altura_real_cm: int):
    # ... (tu función sin cambios)
    print("Paso 3: Deformación 3D con 'smplx' y LBS Robusto...")
    final_model_path = os.path.join(job_dir, "deformed_model_final.obj")
    try:
        base_obj_path = 'backend/assets/smpl/SMPL_NEUTRAL.obj'
        smpl_model_dir = r"D:\Desarrollo\Facultad\SIS330\AplicacionDeCreacionRutinasGYM - copia\gym-ai-app\backend\assets\smpl\SMPL_NEUTRAL.pkl"
        
        mesh = trimesh.load(base_obj_path, force='mesh')
        model = smplx.create(model_path=smpl_model_dir, model_type='smpl', gender='neutral')

        v_template = model.v_template.detach().cpu().numpy().squeeze()
        weights = model.lbs_weights.detach().cpu().numpy()
        joints_template = model.J_regressor.detach().cpu().numpy() @ v_template
        parents = model.parents.detach().cpu().numpy()
        
        with open(landmarks_path, 'r') as f:
            user_points = np.array([[lm['x'], lm['y'], lm['z']] for lm in json.load(f)])

        SMPL_JOINT_MAP = { 'left_hip': 1, 'right_hip': 2, 'left_shoulder': 16, 'right_shoulder': 17, 'left_knee': 4, 'right_knee': 5, 'left_ankle': 7, 'right_ankle': 8, 'left_elbow': 18, 'right_elbow': 19, 'left_wrist': 20, 'right_wrist': 21 }
        bone_scale_factors = {}
        print("DEBUG: Calculando factores de escala de los huesos (proporcional)...")
        for start_name, end_name in BONES_TO_PROCESS:
            user_len = np.linalg.norm(user_points[MP_JOINT_MAP[start_name]] - user_points[MP_JOINT_MAP[end_name]])
            model_len = np.linalg.norm(joints_template[SMPL_JOINT_MAP[start_name]] - joints_template[SMPL_JOINT_MAP[end_name]])
            if model_len < 1e-6: continue
            factor = user_len / model_len
            bone_scale_factors[(SMPL_JOINT_MAP[start_name], SMPL_JOINT_MAP[end_name])] = factor
            print(f"  - Hueso {start_name}->{end_name}, Factor: {factor:.2f}")

        new_joints = np.copy(joints_template)
        for j_idx in range(1, len(parents)):
            parent_idx = parents[j_idx]
            scale = bone_scale_factors.get((parent_idx, j_idx), 1.0)
            bone_vec = joints_template[j_idx] - joints_template[parent_idx]
            new_joints[j_idx] = new_joints[parent_idx] + bone_vec * scale

        joint_displacements = new_joints - joints_template
        vertex_displacements = weights @ joint_displacements
        
        v_proportional = v_template + vertex_displacements
        
        ymin, ymax = v_proportional[:, 1].min(), v_proportional[:, 1].max()
        altura_modelo_actual_m = ymax - ymin
        altura_real_usuario_m = altura_real_cm / 100.0
        
        final_scale_factor = altura_real_usuario_m / altura_modelo_actual_m
        
        print(f"DEBUG: Altura del modelo pre-escalado (m): {altura_modelo_actual_m:.2f}")
        print(f"DEBUG: Factor de escala FINAL calculado: {final_scale_factor:.2f}")
        
        v_deformed_final = v_proportional * final_scale_factor
        
        mesh.vertices = v_deformed_final
        mesh.fix_normals()
        mesh.export(final_model_path)
        print(f"¡Modelo 3D final deformado y guardado en: {final_model_path}!")
        return {"status": "Deformación final con smplx completada."}

    except Exception:
        print("¡¡¡ERROR GRAVE DURANTE LA DEFORMACIÓN CON SMPLX!!!")
        traceback.print_exc()
        return None
# Agrégala junto a tus otras funciones auxiliares
reporte_postural = {}
def analizar_postura_con_landmarks(landmarks: list) -> dict:
    """
    Analiza los landmarks 3D de MediaPipe para detectar posibles
    desbalances posturales.
    """
    reporte_postural = {
        "inclinacion_hombros": "Nivelados",
        "postura_hombros": "Neutra",
        "inclinacion_pelvica": "Neutra"
    }
    
    # Landmarks relevantes
    hombro_izq = landmarks[MP_JOINT_MAP["left_shoulder"]]
    hombro_der = landmarks[MP_JOINT_MAP["right_shoulder"]]
    cadera_izq = landmarks[MP_JOINT_MAP["left_hip"]]
    cadera_der = landmarks[MP_JOINT_MAP["right_hip"]]
    
    # 1. Analizar inclinación de hombros (plano vertical Y)
    # Un umbral pequeño para evitar falsos positivos por la pose
    if abs(hombro_izq['y'] - hombro_der['y']) > 0.03: 
        if hombro_izq['y'] > hombro_der['y']:
            reporte_postural["inclinacion_hombros"] = "Izquierdo más alto"
        else:
            reporte_postural["inclinacion_hombros"] = "Derecho más alto"

    # 2. Analizar postura de hombros (plano sagital Z)
    # Promediamos la profundidad de hombros vs caderas
    z_hombros_avg = (hombro_izq['z'] + hombro_der['z']) / 2
    z_caderas_avg = (cadera_izq['z'] + cadera_der['z']) / 2
    if z_hombros_avg < (z_caderas_avg - 0.05): # Umbral de 5cm
         reporte_postural["postura_hombros"] = "Tendencia a hombros adelantados (cifosis)"
         
    # Puedes añadir más análisis aquí (inclinación pélvica, etc.)

    print("🔎 Reporte postural generado a partir de landmarks.")
    return reporte_postural
def run_ai_pipeline(job_id: str):
    # ... (código existente, solo modificamos la sección final)
    print(f"Iniciando pipeline de IA para el trabajo: {job_id}")
    job_dir = os.path.join("jobs", job_id)
    front_image_path = os.path.join(job_dir, "front_image.jpg")
    side_image_path = os.path.join(job_dir, "side_image.jpg")
    landmarks_output_path = os.path.join(job_dir, "pose_3d_landmarks.json")
    
    draw_landmarks_on_image(front_image_path, os.path.join(job_dir, "front_image_landmarks.jpg"))
    draw_landmarks_on_image(side_image_path, os.path.join(job_dir, "side_image_landmarks.jpg"))
    
    results_front, results_side = None, None
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True) as pose:
        image_front = cv2.imread(front_image_path)
        if image_front is not None:
            results_front = pose.process(cv2.cvtColor(image_front, cv2.COLOR_BGR2RGB))

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True) as pose:
        image_side = cv2.imread(side_image_path)
        if image_side is not None:
            results_side = pose.process(cv2.cvtColor(image_side, cv2.COLOR_BGR2RGB))

    if not (results_front and results_front.pose_world_landmarks and results_side and results_side.pose_world_landmarks):
        print(f"No se detectaron landmarks en la imagen frontal o de perfil.")
        return
        
    print("¡Landmarks detectados!")
    
    landmarks_front = results_front.pose_world_landmarks.landmark
    landmarks_side = results_side.pose_world_landmarks.landmark
    fused_landmarks = [{'x': landmarks_front[i].x, 'y': landmarks_front[i].y, 'z': landmarks_side[i].z} for i in range(len(landmarks_front))]
            
    with open(landmarks_output_path, 'w') as f: json.dump(fused_landmarks, f, indent=4)
    print(f"Landmarks 3D fusionados y guardados.")
    reporte_postural = analizar_postura_con_landmarks(fused_landmarks)

    db = database.SessionLocal()
    try:
        job = db.query(models.AnalysisJob).filter(models.AnalysisJob.job_id == job_id).first()
        if job:
            cuestionario_data = json.loads(job.cuestionario_json)
            altura_real_usuario = cuestionario_data['altura']
            
            arquetipo_detectado = asignar_arquetipo(cuestionario_data)
            print(f"🧬 Arquetipo detectado (v1 Heurística): {arquetipo_detectado}")
            
            # --- 1. CARGAR HISTORIAL DE FEEDBACK (NUEVO) ---
            feedback_history = []
            if job.results_json: # Solo si ya hay resultados guardados
                try:
                    resultados_pasados = json.loads(job.results_json)
                    # Usamos .get() para evitar errores si la clave no existe
                    feedback_history = resultados_pasados.get("feedback_history", []) 
                    print(f"📜 Historial de feedback cargado ({len(feedback_history)} entradas).")
                except json.JSONDecodeError:
                    print("ADVERTENCIA: No se pudo cargar el historial de feedback (JSON corrupto?).")
                    feedback_history = [{"error": "Failed to parse previous feedback"}]
            
            resultado_deformacion = analizar_y_deformar_modelo_smplx(job_dir, landmarks_output_path, altura_real_usuario)
            
            if resultado_deformacion:
                final_model_path = os.path.join(job_dir, "deformed_model_final.obj")
                smpl_model_dir = r"D:\Desarrollo\Facultad\SIS330\AplicacionDeCreacionRutinasGYM - copia\gym-ai-app\backend\assets\smpl\SMPL_NEUTRAL.pkl"

                reporte_3d = generar_reporte_analisis_3d(final_model_path, smpl_model_dir)
                somatotipo_detectado = clasificar_somatotipo(reporte_3d, cuestionario_data)
                print(f"🧬 Somatotipo detectado (v1 Heurística): {somatotipo_detectado}")
                # --- INICIO DE CÓDIGO AÑADIDO ---
                # (Llamamos a Meshy DESPUÉS de generar el reporte 3D,
                # para no bloquear el pipeline principal si Meshy falla)
                print("\n--- Iniciando Tarea de Meshy AI (Paralelo) ---")
                try:
                    # Llama a la nueva función de Meshy usando la imagen frontal
                    meshy_model_url = generar_modelo_3d_con_meshy(
                        image_path=front_image_path, # Usa la imagen frontal
                        job_dir=job_dir,
                        job_id=job_id
                    )
                    if meshy_model_url:
                        print(f"✅ URL del modelo Meshy generada: {meshy_model_url}")
                    else:
                        print("⚠️ No se pudo generar el modelo de Meshy.")
                except Exception as e_meshy:
                    print(f"❌ ERROR FATAL en la integración de Meshy: {e_meshy}")
                    traceback.print_exc()
                print("--- Fin Tarea de Meshy AI ---\n")
                # --- FIN DE CÓDIGO AÑADIDO ---
                
                
                if reporte_3d:
                    reporte_path = os.path.join(job_dir, "reporte_analisis_3d.json")
                    with open(reporte_path, 'w') as f:
                        json.dump(reporte_3d, f, indent=4)
                    
                    print("--- Reporte Final de Análisis 3D ---")
                    print(json.dumps(reporte_3d, indent=4))
                    
                    rutina_generada_texto = llamar_a_gemini(reporte_3d, 
                                                            cuestionario_data,
                                                            reporte_postural,
                                                            somatotipo_detectado,
                                                            arquetipo_detectado,
                                                            feedback_history)
                    # --- INICIO: BLOQUE MODIFICADO ---
                    rutina_data_enriquecida = {}
                        
                    try:
                        # 1. Parsea el JSON de Gemini
                        rutina_data_base = json.loads(rutina_generada_texto)
                            
                        # 2. Llama a la nueva función para añadir GIFs
                        rutina_data_enriquecida = enriquecer_rutina_con_gifs(rutina_data_base)
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ ERROR: Gemini no devolvió un JSON válido. No se pueden añadir GIFs. Error: {e}")
                        # Guardamos el texto crudo para debug
                        rutina_data_enriquecida = {
                            "error_parsing_gemini": True,
                            "raw_text": rutina_generada_texto
                         }
                    except Exception as e_enr:
                            print(f"❌ ERROR: Falló el enriquecimiento de GIFs: {e_enr}")
                            # Si el enriquecimiento falla, al menos guarda la rutina base
                            print("❗️ Guardando la rutina SIN GIFs.")
                            if not rutina_data_enriquecida: # Si aún está vacío
                                try:
                                    rutina_data_enriquecida = json.loads(rutina_generada_texto)
                                except:
                                    rutina_data_enriquecida = {"error_fatal": str(e_enr)}

                    
                    # --- FIN: BLOQUE MODIFICADO ---
                    rutina_path = os.path.join(job_dir, "rutina_generada.json")
                    with open(rutina_path, 'w', encoding='utf-8') as f:
                        json.dump(rutina_data_enriquecida, f, indent=4, ensure_ascii=False)
                    print(f"Rutina guardada en: {rutina_path}")

                    resultados_finales = {
                                    "reporte_3d": reporte_3d,
                                    "reporte_postural": reporte_postural,
                                    "somatotipo": somatotipo_detectado,
                                    "arquetipo": arquetipo_detectado, # (Añade esto si ya lo tienes)
                        
                                    "rutina_generada": rutina_data_enriquecida,
                                    "feedback_history": feedback_history
                    }
                    
                    
                    resultados_limpios = clean_numpy_types(resultados_finales)
                    # --- AÑADE ESTAS 4 LÍNEAS DE PRUEBA ---
                    print("\n--- PRUEBA DE GUARDADO (job.results_json) ---")
                    print("Viendo el primer ejercicio que se va a guardar:")
                    print(json.dumps(resultados_limpios["rutina_generada"]["plan_semanal"][0]["ejercicios"][0], indent=2))
                    print("--- FIN PRUEBA ---\n")
                    # --- FIN DE LÍNEAS DE PRUEBA ---
                    job.status = "completed"
                    job.results_json = json.dumps(resultados_limpios)
                    db.commit()
    finally:
        db.close()

    print(f"Pipeline de IA para el trabajo {job_id} completado.")




def llamar_a_gemini(reporte_3d: dict, cuestionario: dict,reporte_postural: dict,somatotipo_detectado: str, arquetipo_detectado: str,feedback_history: list) -> str:
    print("🧠 Llamando a la API de Gemini para generar la rutina...")
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    feedback_str = "Aún no hay feedback."
    if feedback_history:
        # Formatea el historial para que sea claro en el prompt
        # Tomamos solo los últimos 3 feedbacks para no exceder límites
        formatted_feedback = []
        for entry in feedback_history[-3:]:
            fecha = entry.get("fecha", "Fecha desconocida")
            texto = entry.get("texto_original", "Texto no disponible")
            analisis = entry.get("analisis_ia", {})
            sentimiento_general = analisis.get("sentimiento_general", "N/A")
            entidades = analisis.get("entidades_clave", [])
            entidades_str = ", ".join([f"{e.get('entidad', '?')} ({e.get('aspecto', '?')}: {e.get('sentimiento', '?')})" for e in entidades])
            
            formatted_feedback.append(
                f"- Fecha: {fecha}\n"
                f"  Usuario dijo: \"{texto}\"\n"
                f"  Análisis IA: Sentimiento={sentimiento_general}. Entidades clave: [{entidades_str}]"
            )
        feedback_str = "\n".join(formatted_feedback)

    model = genai.GenerativeModel('models/gemini-2.5-pro') # O el que uses
    
    
    
    prompt_datos = f"""
    Actúa como un entrenador personal experto y científico del deporte. Tu tarea es generar una rutina de gimnasio detallada, segura y altamente personalizada para un usuario.

    **Contexto del Usuario:**
    La siguiente información fue obtenida de un análisis corporal 3D y un formulario. Debes analizar e interpretar estos datos para fundamentar la rutina.

    **1. Datos del Análisis Corporal 3D:**
    {json.dumps(reporte_3d, indent=4)}
    **2. Datos del Análisis Postural (desde MediaPipe):**
    {json.dumps(reporte_postural, indent=4)}
    **3. Clasificación Física (IA v1):**
    * **Somatotipo Inferido:** {somatotipo_detectado}
    * **Arquetipo de Estilo de Vida:** {arquetipo_detectado}
    **4. Datos del Formulario del Usuario:**
    * **Objetivo Principal:** {cuestionario.get('objetivo_principal', 'No especificado')}
    * **Nivel de Experiencia:** {cuestionario.get('nivel_entrenamiento', 'No especificado')}
    * **Días de Entrenamiento por Semana:** {cuestionario.get('frecuencia_entrenamiento', 'No especificado')}
    * **Limitaciones o Lesiones:** {cuestionario.get('lesion_detalle', 'Ninguna')}
    **5. (NUEVO Y CRÍTICO) Feedback Reciente del Usuario:**
        {feedback_str}
    **Instrucciones para la Generación de la Rutina:**

    1.  **Análisis Integral:** Comienza con un breve párrafo analizando **TODOS** los datos disponibles: análisis 3D, postura, somatotipo, arquetipo de estilo de vida, datos del formulario Y el **feedback reciente del usuario** (si existe). Menciona cómo la rutina abordará los puntos clave (desbalances, objetivos, limitaciones, feedback).

    2.  **ADAPTACIÓN BASADA EN FEEDBACK (MUY IMPORTANTE):** Antes de diseñar la rutina, revisa cuidadosamente la sección "**5. Feedback Reciente del Usuario**".
        * Si el feedback menciona **dolor**, **molestia** o **incomodidad** con un ejercicio específico o en una zona corporal durante un ejercicio: **DEBES** reemplazar ese ejercicio por una alternativa más segura que trabaje músculos similares pero evite el estrés en la zona afectada (ej. cambiar 'Press Militar con Barra' por 'Press Arnold con Mancuernas' si hay dolor de hombro). Si no es posible reemplazarlo, modifica el ejercicio (ej. reducir peso significativamente, ajustar el rango de movimiento, cambiar el agarre) o elimínalo temporalmente. **Justifica claramente este cambio en la sección de justificación del ejercicio modificado.**
        * Si el feedback menciona **disfrute**, **buenas sensaciones** (ej. "buen bombeo", "me encantó") con un tipo de ejercicio o grupo muscular, considera mantener o reforzar ese aspecto en la nueva rutina, siempre que sea coherente con los objetivos generales.
        * Si el feedback menciona explícitamente que la rutina anterior fue **demasiado fácil** o **demasiado difícil** en general, ajusta sutilmente el volumen total (número de series) o la intensidad sugerida (ej. RPE, %1RM si aplica, o simplemente el rango de repeticiones) para la nueva semana.

    3.  **Estructura de la Rutina:** Diseña un plan de entrenamiento semanal completo, dividido según los días indicados por el usuario.

    4.  **Detalle por Ejercicio:** Para cada ejercicio, especifica claramente:
        * Nombre del Ejercicio(en español).
        * **nombre_en_ingles:** El nombre exacto del ejercicio en INGLÉS. **IMPORTANTE:** Usa nombres estándar de la base de datos de ExerciseDB. (Ej. 'barbell full squat', 'dumbbell bench press', 'cable crossover', 'lying leg curl'). Evita nombres coloquiales como "skullcrushers".
        * Repeticiones (o tiempo).
        * Tiempo de Descanso entre series.
        * **Justificación (Por qué):** Explica (1-2 frases) CÓMO ayuda al objetivo Y POR QUÉ se eligió (considerando datos 3D, postura, feedback, etc.). **Si este ejercicio fue modificado debido al feedback, explícalo aquí.**
        * **Técnica y Propósito:** Breve nota sobre la forma correcta y su rol en la rutina.

    5.  **Principios de Entrenamiento:** Asegúrate de que la rutina siga los principios de sobrecarga progresiva (explica brevemente cómo progresar), especificidad e individualización.

    6.  **Seguridad General:** Considera las lesiones preexistentes mencionadas en el formulario y el análisis postural. Asegúrate de que la rutina sea equilibrada para prevenir nuevas lesiones (ej. incluye trabajo de cadena posterior si el usuario es "Trabajador de Oficina").
    **INSTRUCCIONES DE SALIDA (MUY IMPORTANTE):**
    Responde **ÚNICAMENTE** con un objeto JSON válido. No incluyas "```json" ni ningún
    texto explicativo fuera del JSON.
    
    El formato JSON DEBE ser el siguiente:

    """
    prompt_formato = """
    {
      "resumen_ia": "Analicé tu feedback y somatotipo. Hice estos cambios...",
      "plan_semanal": [
        {
          "dia": 1,
          "titulo": "Tren Superior - Empuje (Fuerza)",
          "ejercicios": [
            {
              "nombre": "Press de Banca con Mancuernas",
              "nombre_en_ingles": "Dumbbell Bench Press",
              "series": 4,
              "repeticiones": "6-8",
              "descanso": "90-120 seg",
              "justificacion": "Priorizamos mancuernas para abordar tu desbalance de volumen...",
              "tecnica": "Controla la bajada y empuja de forma explosiva."
            },
            {
              "nombre": "Press Militar de Pie",
              "nombre_en_ingles": "Barbell Standing Overhead Press",
              "series": 3,
              "repeticiones": "8-10",
              "descanso": "90 seg",
              "justificacion": "Ejercicio fundamental para fuerza de hombro y core.",
              "tecnica": "Aprieta glúteos y abdomen."
            }
          ]
        },
        {
          "dia": 2,
          "titulo": "Tren Inferior (Hipertrofia)",
          "ejercicios": [
            {
              "nombre": "Sentadilla Trasera",
              "nombre_en_ingles": "barbell full squat",
              "series": 4,
              "repeticiones": "8-12",
              "descanso": "120 seg",
              "justificacion": "El mejor ejercicio para construir masa en las piernas.",
              "tecnica": "Mantén la espalda recta y rompe el paralelo."
            }
          ]
        }
      ]
    }
    """
    prompt_template = prompt_datos + prompt_formato
    try:
        response = model.generate_content(prompt_template)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        # --- ¡¡¡AÑADE ESTAS 4 LÍNEAS DE DEBUG!!! ---
        print("\n--- DEBUG: RESPUESTA CRUDA DE GEMINI ---")
        print(response.text)
        print("--- FIN DEBUG: RESPUESTA CRUDA ---\n")
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if not match:
            raise ValueError("No se encontró un objeto JSON válido en la respuesta de la IA.")
            
        json_text = match.group(0)
        # --- FIN DEBUG ---
        json.loads(json_text) # Intenta parsearlo para asegurar que es válido
        print("✅ Rutina generada por Gemini exitosamente.")
        print("Somatotipo detectado:", somatotipo_detectado)
        return json_text
    except Exception as e:
        print(f"❌ ERROR al llamar a la API de Gemini: {e}")
        traceback.print_exc() # Imprime el error completo para más detalles
        return "Error al generar la rutina con la IA. Por favor, intente más tarde."


# Enriquecimiento de rutina con GIFs


def enriquecer_rutina_con_gifs(rutina_data: dict) -> dict:
    """
    Recorre el JSON de la rutina y añade la 'gifUrl' a cada ejercicio
    llamando a la API de ExerciseDB. (VERSIÓN SEGURA)
    """
    print("🏋️‍♂️ Iniciando enriquecimiento de rutina con GIFs...")
    
    if not RAPIDAPI_KEY:
        print("❌ ERROR: No hay RAPIDAPI_KEY. Saltando enriquecimiento de GIFs.")
        return rutina_data 

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "exercisedb.p.rapidapi.com"
    }
    url_base_busqueda = "https://exercisedb.p.rapidapi.com/exercises/name/"
    url_base_gif = "https://exercisedb.p.rapidapi.com/image"

    try:
        # Usamos .get() para evitar errores si las claves no existen
        plan_semanal = rutina_data.get("plan_semanal", [])
        if not isinstance(plan_semanal, list):
            print("  - ADVERTENCIA: 'plan_semanal' no es una lista. Saltando.")
            return rutina_data

        for dia in plan_semanal:
            # Aseguramos que 'dia' sea un diccionario
            if not isinstance(dia, dict):
                print("  - ADVERTENCIA: 'dia' no es un diccionario. Saltando.")
                continue

            ejercicios_lista = dia.get("ejercicios", [])
            if not isinstance(ejercicios_lista, list):
                print(f"  - ADVERTENCIA: 'ejercicios' en {dia.get('titulo')} no es una lista. Saltando.")
                continue

            for ejercicio in ejercicios_lista:
                # Aseguramos que 'ejercicio' sea un diccionario
                if not isinstance(ejercicio, dict):
                    print("  - ADVERTENCIA: 'ejercicio' no es un diccionario. Saltando.")
                    continue
                
                # Seteamos un valor por defecto
                ejercicio["exerciseId"] = None # Seteamos un valor por defecto
                nombre_ingles = ejercicio.get("nombre_en_ingles")
                
                if not nombre_ingles:
                    print(f"  - Advertencia: '{ejercicio.get('nombre')}' no tiene 'nombre_en_ingles'.")
                    continue
                
                try:
                    url_busqueda_completa = url_base_busqueda + quote(nombre_ingles.lower())
                    response_busqueda = requests.get(url_busqueda_completa, headers=headers)
                    response_busqueda.raise_for_status() 
                    
                    resultados = response_busqueda.json()
                    
                    if resultados and isinstance(resultados, list) and len(resultados) > 0:
                        exercise_id = resultados[0].get("id")
                        if exercise_id:
                            ejercicio["exerciseId"] = exercise_id
                            # <-- MODIFICACIÓN
                            print(f"  - ID encontrado para: {nombre_ingles} (ID: {exercise_id})")
                        else:
                            print(f"  - No se encontró ID para: {nombre_ingles}")
                    else:
                        print(f"  - No se encontró ID para: {nombre_ingles}")
                
                except requests.RequestException as e_req:
                    print(f"  - ❌ ERROR API para '{nombre_ingles}': {e_req}")
                except Exception as e_inner:
                    print(f"  - ❌ ERROR procesando '{nombre_ingles}': {e_inner}")

        print("✅ Enriquecimiento de GIFs completado.")
        return rutina_data # <-- Devuelve el objeto modificado

    except Exception as e_fatal:
        print(f"❌ ERROR FATAL durante el enriquecimiento de GIFs: {e_fatal}")
        traceback.print_exc() # Imprime el error completo
        return rutina_data # Devuelve el objeto (posiblemente a medio modificar)


# clasificación de somatotipo
def clasificar_somatotipo(reporte_3d: dict, cuestionario: dict) -> str:
    """
    Clasifica el somatotipo basado en heurísticas del reporte 3D y el formulario.
    Esta es una v1; en v2 puede ser reemplazado por un modelo SVM o de Red Neuronal.
    """
    try:
        # Extraer métricas clave
        # Usamos get() para evitar errores si una clave no existe
        ica = reporte_3d.get("indices_corporales", {}).get("indice_cintura_altura", 0)
        icc = reporte_3d.get("indices_corporales", {}).get("indice_cintura_cadera", 0)
        peso = cuestionario.get("peso_actual", 70)
        altura_cm = reporte_3d.get("altura_cm", cuestionario.get("altura", 170))
        if altura_cm == 0: altura_cm = 170 # Evitar división por cero

        imc = peso / ((altura_cm / 100) ** 2)

        # --- Lógica de Clasificación Heurística ---
        # (Estos umbrales son ejemplos, puedes ajustarlos con un experto)

        # Fuerte indicador de Endomorfo: alto % de grasa (inferido de ICA/ICC/IMC)
        if ica > 0.55 or icc > 0.95 or imc > 28:
            return "Endomorfo"

        # Fuerte indicador de Ectomorfo: bajo % de grasa y estructura delgada
        if imc < 19 and ica < 0.45:
            return "Ectomorfo"

        # Mesomorfo: intermedio, a menudo con buen IMC pero bajo ICA/ICC
        if 20 < imc < 27 and ica < 0.5 and icc < 0.9:
            return "Mesomorfo"

        # Casos por defecto
        if imc > 25: return "Endo-Mesomorfo"
        if imc < 20: return "Ecto-Mesomorfo"

        return "Mesomorfo" # Default

    except Exception as e:
        print(f"ADVERTENCIA: Falló la clasificación de somatotipo: {e}")
        return "No clasificado"



# Arquetipo de estilo de vida
def asignar_arquetipo(cuestionario: dict) -> str:
    """
    Asigna un arquetipo de estilo de vida basado en heurísticas del formulario (v1).
    """
    try:
        estres = cuestionario.get("nivel_estres", "moderado").lower()
        sueño = cuestionario.get("horas_sueño", "7-8").lower()
        trabajo = cuestionario.get("tipo_trabajo", "sedentario").lower()
        frecuencia = cuestionario.get("frecuencia_entrenamiento", "3-4").lower()

        if "alto" in estres and "sedentario" in trabajo and ("1-2" in frecuencia or "3-4" in frecuencia):
            return "Ejecutivo Ocupado (Alto estrés, sedentario, tiempo limitado)"
        if "menos de 6" in sueño and "alto" in estres:
            return "Estudiante/Trabajador Nocturno (Estrés alto, falta de sueño)"
        if "bajo" in estres and "activo" in trabajo and "5+" in frecuencia:
            return "Entusiasta del Fitness (Alta dedicación, bajo estrés)"
        if "sedentario" in trabajo:
            return "Trabajador de Oficina (Riesgo postural, sedentario)"

        return "Estilo de Vida Equilibrado"
    except Exception as e:
        print(f"ADVERTENCIA: Falló la asignación de arquetipo: {e}")
        return "No clasificado"


# --- 5. Endpoints de la API ---
@app.post("/analysis")
def start_analysis(
    # --- (Todos tus parámetros (background_tasks, current_user, etc.)
    # ---  permanecen exactamente igual que antes) ---
    background_tasks: BackgroundTasks, 
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    regenerate_job_id: Optional[str] = Form(None), 
    nivel_entrenamiento: Optional[str] = Form(None),
    frecuencia_entrenamiento: Optional[str] = Form(None),
    objetivo_principal: Optional[str] = Form(None),
    genero: Optional[str] = Form(None),
    fecha_nacimiento: Optional[date] = Form(None),
    altura: Optional[int] = Form(None),
    peso_actual: Optional[float] = Form(None), 
    lesiones: Optional[str] = Form(None), 
    lesion_detalle: Optional[str] = Form(None),
    horas_sueño: Optional[str] = Form("7-8 horas"),
    nivel_estres: Optional[str] = Form("Moderado"),
    tipo_trabajo: Optional[str] = Form("Sedentario (oficina)"),
    front_image: Optional[UploadFile] = File(None),
    side_image: Optional[UploadFile] = File(None),
    back_image: Optional[UploadFile] = File(None),
):

    # --- 3. ESTA ES LA LÓGICA DE REGENERACIÓN (MODIFICADA) ---
    if regenerate_job_id:
        print(f"♻️ Solicitud de REGENERACIÓN (nuevo job) recibida de: {current_user.email} basado en job: {regenerate_job_id}")

        # 1. Encuentra el job "padre"
        old_job = db.query(models.AnalysisJob).filter(
                models.AnalysisJob.job_id == regenerate_job_id,
                models.AnalysisJob.owner_id == current_user.id
            ).first()

        if not old_job:
            raise HTTPException(status_code=404, detail="Job 'padre' no encontrado o no te pertenece.")

        # 2. Extrae el historial de feedback del job "padre"
        feedback_history = []
        if old_job.results_json:
            try:
                old_results = json.loads(old_job.results_json)
                feedback_history = old_results.get("feedback_history", [])
            except json.JSONDecodeError:
                pass # El feedback se queda vacío si el JSON está mal

        # 3. Crea un NUEVO job
        new_job = models.AnalysisJob(
            # Copia los datos del job "padre"
            owner_id=current_user.id,
            cuestionario_json=old_job.cuestionario_json, 
            # Inserta el historial de feedback en el results_json del *nuevo* job
            results_json=json.dumps({"feedback_history": feedback_history}) 
        )

        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        # 4. Crea las carpetas y COPIA las imágenes del job "padre"
        old_job_dir = os.path.join("jobs", old_job.job_id)
        new_job_dir = os.path.join("jobs", new_job.job_id)
        os.makedirs(new_job_dir, exist_ok=True)

        try:
            shutil.copyfile(
                os.path.join(old_job_dir, "front_image.jpg"),
                os.path.join(new_job_dir, "front_image.jpg")
            )
            shutil.copyfile(
                os.path.join(old_job_dir, "side_image.jpg"),
                os.path.join(new_job_dir, "side_image.jpg")
            )
            shutil.copyfile(
                os.path.join(old_job_dir, "back_image.jpg"),
                os.path.join(new_job_dir, "back_image.jpg")
            )
        except FileNotFoundError:
            print(f"ADVERTENCIA: No se encontraron imágenes para copiar del job {old_job.job_id}")
            # El pipeline podría fallar aquí si las imágenes son 100% necesarias

        # 5. Lanza el pipeline para el NUEVO job
        background_tasks.add_task(run_ai_pipeline, new_job.job_id)

        # 6. Devuelve el NUEVO job_id
        return {"job_id": new_job.job_id, "status": new_job.status}

    
# --- 4. LÓGICA DE NUEVO JOB (CON CORRECCIÓN) ---
    else:
        print(f"🌱 Nuevo análisis solicitado por: {current_user.email}")
        
        # 1. Validación de campos requeridos
        required_fields = [
            nivel_entrenamiento, frecuencia_entrenamiento, objetivo_principal, 
            genero, fecha_nacimiento, altura, peso_actual, lesiones, 
            front_image, side_image, back_image
        ]
        if not all(required_fields):
            raise HTTPException(status_code=422, detail="Faltan campos requeridos para un nuevo análisis.")

        # 2. Validación del Cuestionario Schema
        # (Usamos .dict() para excluir los UploadFile que no están en el schema)
        form_data_dict = {
            "nivel_entrenamiento": nivel_entrenamiento,
            "frecuencia_entrenamiento": frecuencia_entrenamiento,
            "objetivo_principal": objetivo_principal,
            "genero": genero,
            "fecha_nacimiento": fecha_nacimiento,
            "altura": altura,
            "peso_actual": peso_actual,
            "lesiones": lesiones,
            "lesion_detalle": lesion_detalle,
            "horas_sueño": horas_sueño,
            "nivel_estres": nivel_estres,
            "tipo_trabajo": tipo_trabajo,
        }
        try:
            cuestionario = schemas.CuestionarioSchema(**form_data_dict)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Error de validación del cuestionario: {e}")

        # 3. Creación del Job en la BD
        new_job = models.AnalysisJob(
            cuestionario_json=cuestionario.model_dump_json(),
            owner_id=current_user.id
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        # 4. Creación del directorio
        job_dir = os.path.join("jobs", new_job.job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # 5. Guardado de archivos (CON LA CORRECCIÓN .seek(0))
        try:
            # Rebobina el archivo antes de copiarlo
            front_image.file.seek(0)
            with open(os.path.join(job_dir, "front_image.jpg"), "wb") as buffer:
                shutil.copyfileobj(front_image.file, buffer)
            
            # Rebobina el archivo antes de copiarlo
            side_image.file.seek(0)
            with open(os.path.join(job_dir, "side_image.jpg"), "wb") as buffer:
                shutil.copyfileobj(side_image.file, buffer)
            
            # Rebobina el archivo antes de copiarlo
            back_image.file.seek(0)
            with open(os.path.join(job_dir, "back_image.jpg"), "wb") as buffer:
                shutil.copyfileobj(back_image.file, buffer)
        
        except Exception as e:
            # Si algo falla al guardar archivos, informa el error
            print(f"--- !!! ERROR AL GUARDAR ARCHIVOS !!! ---: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error al guardar imágenes: {e}")

        # 6. Lanzar tarea en segundo plano
        background_tasks.add_task(run_ai_pipeline, new_job.job_id)
        
        # 7. Devolver respuesta
        return {"job_id": new_job.job_id, "status": new_job.status}



@app.get("/analysis/status/{job_id}")
def get_analysis_status(job_id: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    try:
        print(f"\n--- DEBUG (1/5): Endpoint /analysis/status/{job_id} HIT ---")
        job = db.query(models.AnalysisJob).filter(
                models.AnalysisJob.job_id == job_id,
                models.AnalysisJob.owner_id == current_user.id
            ).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        results_data = None
        
        if job.status == "completed" and job.results_json:
            print("--- DEBUG (2/5): Job completo. Intentando cargar JSON... ---")
            
            # Intenta cargar el JSON
            try:
                raw_results = json.loads(job.results_json)
                print("--- DEBUG (3/5): JSON cargado. Limpiando tipos de Numpy... ---")
            except Exception as e_load:
                print(f"--- !!! ERROR (LOAD) !!!: json.loads FALLÓ: {e_load} ---")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to load results: {e_load}")

            # Intenta limpiar los datos
            try:
                results_data = clean_numpy_types(raw_results)
                print("--- DEBUG (4/5): Tipos de Numpy limpios. ---")
            except Exception as e_clean:
                print(f"--- !!! ERROR (CLEAN) !!!: clean_numpy_types FALLÓ: {e_clean} ---")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to clean results: {e_clean}")

        else:
            print(f"--- DEBUG (2/5): Job status es '{job.status}'. No hay resultados. ---")

        print("--- DEBUG (5/5): Enviando respuesta 200 OK al frontend. ---")
        return {
            "job_id": job.job_id,
            "status": job.status,
            "results_data": results_data
        }
    
    except Exception as e_main:
        # Captura cualquier otro error
        print(f"--- !!! ERROR FATAL (MAIN) !!!: El endpoint falló: {e_main} ---")
        traceback.print_exc() # Imprime el error completo
        raise HTTPException(status_code=500, detail=f"Main error: {e_main}")
    
    
#Modulo feedback
# --- Añade esta nueva función ---
async def analizar_feedback_con_gemini(texto_feedback: str) -> dict:
    """
    Usa Gemini para realizar un Análisis de Sentimiento Basado en Aspectos (ABSA)
    del feedback de entrenamiento.
    """
    print(f"🧠 Analizando feedback con Gemini: '{texto_feedback[:50]}...'")
    model = genai.GenerativeModel('models/gemini-2.5-pro') # O el modelo que prefieras

    prompt = f"""
    Eres un analizador de texto experto en fitness. Analiza el siguiente comentario de un usuario
    después de una sesión de gimnasio. Tu objetivo es:
    1. Determinar el sentimiento general (positivo, negativo, mixto).
    2. Identificar entidades clave: ejercicios específicos o partes del cuerpo mencionadas.
    3. Para cada entidad, extraer el aspecto discutido (ej. dolor, dificultad, bombeo, técnica)
       y el sentimiento asociado a ESE aspecto.
    4. Extraer la percepción general de dificultad de la sesión si se menciona.

    Responde **ÚNICAMENTE** en formato JSON válido. No incluyas explicaciones adicionales
    ni texto antes o después del JSON.

    Texto del Usuario:
    "{texto_feedback}"

    Ejemplo de Salida JSON Esperada:
    {{
      "sentimiento_general": "mixto-positivo",
      "percepcion_dificultad": "alta",
      "entidades_clave": [
        {{"entidad": "sentadillas", "aspecto": "dolor", "sentimiento": "negativo", "zona_cuerpo": "rodilla", "texto_fragmento": "me dolieron en la rodilla"}},
        {{"entidad": "curl de bíceps", "aspecto": "bombeo", "sentimiento": "muy_positivo", "zona_cuerpo": "bíceps", "texto_fragmento": "bombeo fue increíble"}}
      ]
    }}
    """
    try:
        # Usamos generate_content_async para no bloquear el servidor si Gemini tarda
        response = await model.generate_content_async(prompt)
        # Limpiamos la respuesta para asegurar que sea solo JSON
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        print("✅ Feedback analizado por Gemini.")
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ ERROR al analizar feedback con Gemini: {e}")
        traceback.print_exc()
        # Devolvemos un error estructurado si falla
        return {
            "sentimiento_general": "error",
            "percepcion_dificultad": "desconocida",
            "entidades_clave": [],
            "error_message": f"Fallo en análisis IA: {str(e)}"
        }
        
# --- Añade este nuevo endpoint ---
@app.post("/feedback/{job_id}")
async def recibir_feedback( # <-- Hazlo async
    job_id: str,
    feedback_text: str = Form(...), # Recibe el texto del formulario del frontend
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    print(f"📥 Feedback recibido de: {current_user.email} para job: {job_id}")
    job = db.query(models.AnalysisJob).filter(
            models.AnalysisJob.job_id == job_id,
            models.AnalysisJob.owner_id == current_user.id
        ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 1. Llama a la función de IA para analizar el texto (espera el resultado)
    reporte_sentimiento = await analizar_feedback_con_gemini(feedback_text)

    # 2. Carga los resultados actuales (si existen)
    try:
        # Asegúrate de que results_json no sea None antes de cargar
        resultados_actuales = json.loads(job.results_json) if job.results_json else {}
    except json.JSONDecodeError:
        # Si el JSON guardado está corrupto, empezamos de nuevo
        resultados_actuales = {"error_previo": "JSON corrupto en DB"}

    # 3. Inicializa o añade al historial de feedback
    if "feedback_history" not in resultados_actuales or not isinstance(resultados_actuales.get("feedback_history"), list):
        resultados_actuales["feedback_history"] = []

    # Añade el nuevo feedback (texto original + análisis IA)
    resultados_actuales["feedback_history"].append({
        "fecha": date.today().isoformat(), # Guarda la fecha del feedback
        "texto_original": feedback_text,
        "analisis_ia": reporte_sentimiento
    })

    # 4. Limpia ANTES de guardar (por si acaso) y guarda el JSON actualizado
    try:
        resultados_limpios = clean_numpy_types(resultados_actuales) # Limpia todo el objeto
        job.results_json = json.dumps(resultados_limpios)
        db.commit()
        print(f"💾 Feedback guardado y análisis IA completado para job: {job_id}")
    except Exception as e_save:
        print(f"❌ ERROR al guardar feedback en DB: {e_save}")
        db.rollback() # Deshace cambios si falla el guardado
        raise HTTPException(status_code=500, detail="Error al guardar el feedback.")

    # Devuelve el análisis al frontend para confirmación (opcional)
    return {"status": "feedback recibido y analizado", "analisis": reporte_sentimiento}

# --- AÑADE EL ENDPOINT DE REGISTRO ---
@app.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    print("\n--- DEBUG: Entrando a /register ---")
    try:
        # 1. Verifica si el email ya existe
        print(f"--- DEBUG: Buscando email {user.email} ---")
        db_user = get_user_by_email(db, email=user.email)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El email ya está registrado."
            )

        # 2. Hashea la contraseña
        print("--- DEBUG: Hasheando contraseña... ---")
        hashed_password = auth_utils.get_password_hash(user.password)
        print("--- DEBUG: Contraseña hasheada. ---")

        # 3. Crea el nuevo objeto de usuario
        print("--- DEBUG: Creando objeto models.User... ---")
        new_user = models.User(email=user.email, hashed_password=hashed_password)

        # 4. Guarda en la base de datos
        print("--- DEBUG: Añadiendo a la sesión... ---")
        db.add(new_user)
        print("--- DEBUG: Haciendo db.commit()... ---")
        db.commit() # <-- El error 500 debe estar aquí o en el hash
        print("--- DEBUG: db.commit() exitoso. ---")
        
        db.refresh(new_user)
        print("--- DEBUG: db.refresh() exitoso. ---")

        return new_user

    except Exception as e:
        # --- ¡ESTO ES LO QUE NECESITAMOS VER! ---
        print(f"\n--- !!! ERROR FATAL EN /register !!! ---")
        print(f"Error: {e}")
        # Imprime el traceback completo en tu terminal
        traceback.print_exc() 
        
        # Devuelve un 500 con el mensaje de error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {e}"
        )

# --- AÑADE EL ENDPOINT DE LOGIN (TOKEN) ---
@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    """
    Endpoint para iniciar sesión y obtener un token JWT.
    FastAPI espera un formulario con 'username' y 'password'.
    Usaremos el campo 'username' para nuestro 'email'.
    """
    # 1. Busca al usuario por su email (que viene en form_data.username)
    user = get_user_by_email(db, email=form_data.username)

    # 2. Verifica que el usuario exista y la contraseña sea correcta
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Crea el token de acceso
    # El "subject" ('sub') del token será el email del usuario
    access_token = auth_utils.create_access_token(
        data={"sub": user.email}
    )

    # 4. Devuelve el token
    return {"access_token": access_token, "token_type": "bearer"}

# --- REEMPLAZA TU FUNCIÓN 'get_analysis_history' POR ESTA ---

# 1. Quita el 'response_model' de aquí, ya que devolveremos una lista simple
@app.get("/analysis/history")
def get_analysis_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    print(f"\n--- DEBUG: Entrando a /analysis/history para {current_user.email} ---")
    try:
        print("--- DEBUG: Ejecutando consulta a la BD... ---")
        jobs = db.query(models.AnalysisJob).filter(
            models.AnalysisJob.owner_id == current_user.id
        ).order_by(
            models.AnalysisJob.created_at.desc()
        ).all()
        print(f"--- DEBUG: Consulta exitosa. {len(jobs)} jobs encontrados. ---")
        
        # 2. Construye la respuesta manualmente
        results_list = []
        for job in jobs:
            results_list.append({
                "job_id": job.job_id,
                "status": job.status,
                # 3. Convierte la fecha a un string estándar ISO
                #    Esto es seguro para JSON y el frontend lo puede leer
                "created_at": job.created_at.isoformat() 
            })

        print("--- DEBUG: Devolviendo lista de resultados construida manualmente. ---")
        return results_list # Devuelve la lista de diccionarios simple

    except Exception as e:
        print(f"\n--- !!! ERROR FATAL EN /analysis/history !!! ---")
        print(f"Error: {e}")
        traceback.print_exc() 
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {e}"
        )
        
@app.get("/exercise_gif/{exercise_id}")
async def get_exercise_gif(exercise_id: str):
    """
    Endpoint Proxy: Obtiene el GIF desde RapidAPI y lo retransmite.
    Esto evita el error de CORS en el frontend.
    """
    if not RAPIDAPI_KEY:
        raise HTTPException(status_code=500, detail="El servidor no tiene RAPIDAPI_KEY")

    url = f"https://exercisedb.p.rapidapi.com/image?exerciseId={exercise_id}&resolution=180"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "exercisedb.p.rapidapi.com"
    }

    try:
        # Hacemos la petición con stream=True para no cargar todo el GIF en memoria
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status() # Lanza error si la API falla

        # Retransmitimos la respuesta de RapidAPI directamente al frontend
        return StreamingResponse(
            response.iter_content(chunk_size=8192), 
            media_type=response.headers.get("Content-Type", "image/gif")
        )

    except requests.RequestException as e:
        # Si RapidAPI falla (ej. 404), devuelve un error
        print(f"❌ ERROR en Proxy de GIF (ID: {exercise_id}): {e}")
        raise HTTPException(status_code=e.response.status_code if e.response else 500, detail="Error al obtener el GIF de RapidAPI.")
    except Exception as e_gen:
        print(f"❌ ERROR genérico en Proxy de GIF (ID: {exercise_id}): {e_gen}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar el GIF.")
