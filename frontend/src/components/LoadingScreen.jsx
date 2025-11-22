// frontend/src/components/LoadingScreen.jsx
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function LoadingScreen() {
  const { jobId } = useParams(); // Obtiene el "jobId" de la URL
  const { token , userEmail } = useAuth();
  const navigate = useNavigate();
  const [status, setStatus] = useState('pending');
  const [error, setError] = useState(null);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:8000/analysis/status/${jobId}`, {
          // --- 3. AÑADE ESTE OBJETO 'headers' ---
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (!response.ok) {
          throw new Error('No se pudo verificar el estado del análisis.');
        }
        const result = await response.json();
        setStatus(result.status);

        if (result.status === 'completed') {
          // ¡Éxito! Detenemos el polling y redirigimos a resultados
          navigate(`/results/${jobId}`);
        } else if (result.status === 'failed') {
          // (Suponiendo que tu backend puede tener un estado 'failed')
          setError('El análisis falló. Por favor, intenta de nuevo.');
        }

      } catch (err) {
        setError(err.message);
      }
    };

    // Iniciar el polling inmediatamente al cargar
    pollStatus();

    // Configurar un intervalo para seguir preguntando cada 5 segundos
    const intervalId = setInterval(() => {
      if (status === 'pending') {
        pollStatus();
      }
    }, 5000); // 5 segundos

    // Limpieza: detener el intervalo si el componente se desmonta
    return () => clearInterval(intervalId);

  }, [jobId, status, navigate]); // Dependencias del efecto

  return (
    <div className="loading-container">
      <h2>Estamos analizando tu información...</h2>
      <p>Esto puede tardar uno o dos minutos. No cierres esta página.</p>
      <div className="spinner"></div> {/* (Necesitarás CSS para este spinner) */}
      <p>Estado actual: <strong>{status}</strong></p>
      <p>Generando análisis y rutina para: {userEmail}</p>
      {error && <p className="message message-error">{error}</p>}
    </div>
  );
}

export default LoadingScreen;