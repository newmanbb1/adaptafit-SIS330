// frontend/src/components/HistoryPage.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function HistoryPage() {
  const [jobs, setJobs] = useState([]); // Almacenará la lista de jobs
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const { token } = useAuth(); // Necesitamos el token para la llamada

  useEffect(() => {
    const fetchHistory = async () => {
      if (!token) {
        setIsLoading(false);
        return; // No hacer nada si no hay token
      }

      try {
        setError(null);
        setIsLoading(true);
        const response = await fetch('http://127.0.0.1:8000/analysis/history', {
          headers: {
            'Authorization': `Bearer ${token}`, // <-- La autenticación
          },
        });

        if (!response.ok) {
          throw new Error('No se pudo cargar el historial.');
        }

        const data = await response.json(); // Data es un array de JobHistoryItem
        setJobs(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, [token]); // Se ejecuta cada vez que el token cambia (al iniciar sesión)

  // Función para formatear la fecha
  const formatJobDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // --- Renderizado ---

  if (isLoading) {
    return <div className="loading-container">Cargando historial...</div>;
  }

  if (error) {
    return <div className="message message-error">{error}</div>;
  }

  if (jobs.length === 0) {
    return (
      <div className="history-container">
        <h1>Mis Rutinas</h1>
        <p>Aún no has generado ningún análisis. ¡Empieza creando uno!</p>
        <Link to="/" className="submit-button">Crear mi primer análisis</Link>
      </div>
    );
  }

  return (
    <div className="history-container">
      <h1>Mis Rutinas</h1>
      <p>Aquí están todos los análisis que has generado. Haz clic en uno para ver los resultados.</p>
      <ul className="job-list">
        {jobs.map((job) => (
          <li key={job.job_id} className="job-item">
            <Link to={`/results/${job.job_id}`}>
              <strong>Análisis del {formatJobDate(job.created_at)}</strong>
              <span className={`status-badge status-${job.status}`}>
                {job.status === 'completed' ? 'Completado' : 'Pendiente'}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default HistoryPage;