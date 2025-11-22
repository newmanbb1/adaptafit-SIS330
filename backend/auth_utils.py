# backend/auth_utils.py
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt

# Cargar variables de entorno (para las claves secretas)

# --- CONFIGURACIÓN DE SEGURIDAD ---
#SECRET_KEY = os.getenv("SECRET_KEY")
#ALGORITHM = os.getenv("ALGORITHM")
#ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))


SECRET_KEY="4d1e730051366e9ba40f19e8240fcb940d39eb83271a6e0155a625f3a669ba8b"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=60

#if not SECRET_KEY:
#   raise ValueError("No se encontró SECRET_KEY en el archivo .env")

# Contexto para Hashear Contraseñas (usando bcrypt)
pwd_context = CryptContext(schemes=["sha256_crypt", "bcrypt"], deprecated="auto")
# --- FUNCIONES DE CONTRASEÑA ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica que la contraseña plana coincida con el hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Genera un hash bcrypt de la contraseña."""
    return pwd_context.hash(password)

# --- FUNCIONES DE TOKEN (JWT) ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Crea un nuevo token JWT."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[dict]:
    """
    Decodifica un token. Devuelve el payload (datos) si es válido, 
    o None si no lo es.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None