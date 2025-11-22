# backend/schemas.py
from pydantic import BaseModel, Field,EmailStr
from datetime import date
from typing import Optional,List

# --- ESQUEMAS DE TOKEN ---

class Token(BaseModel):
    """Esquema para la respuesta del token de acceso."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Esquema para los datos contenidos dentro del token."""
    email: Optional[str] = None

# --- ESQUEMAS DE USUARIO ---

class UserBase(BaseModel):
    """Esquema base del usuario, con el email."""
    email: EmailStr # EmailStr de Pydantic valida automáticamente el email

class UserCreate(UserBase):
    """Esquema para crear un usuario nuevo (lo que recibimos en /register)."""
    password: str = Field(..., min_length=8) # Requiere al menos 8 caracteres

class UserLogin(BaseModel):
    """Esquema para iniciar sesión (lo que recibimos en /token)."""
    email: EmailStr
    password: str

class User(UserBase):
    """Esquema para leer un usuario de la BD (lo que devolvemos)."""
    id: str

    # Configuración para que Pydantic funcione con los modelos de SQLAlchemy
    class Config:
        from_attributes = True 
        # (Nota: 'from_attributes' reemplaza a 'orm_mode' en Pydantic v2)

class JobHistoryItem(BaseModel):
    """Esquema para un item individual en la lista del historial."""
    job_id: str
    status: str
    created_at: str # Asegúrate de importar 'datetime' de 'datetime'

    class Config:
        from_attributes = True

class CuestionarioSchema(BaseModel):
    nivel_entrenamiento: str
    frecuencia_entrenamiento: str
    objetivo_principal: str
    genero: str
    fecha_nacimiento: date
    altura: int = Field(..., gt=0) # gt=0 asegura que la altura sea > 0
    peso_actual: float = Field(..., gt=0) # gt=0 asegura que el peso sea > 0
    lesiones: str
    lesion_detalle: Optional[str] = None
# --- ¡ASEGÚRATE DE AÑADIR ESTAS 3 LÍNEAS! ---
    horas_sueño: Optional[str] = None
    nivel_estres: Optional[str] = None
    tipo_trabajo: Optional[str] = None

    class Config:
        from_attributes = True # Añade esto también