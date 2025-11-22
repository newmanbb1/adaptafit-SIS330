// frontend/src/components/ProtectedRoute.jsx
import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function ProtectedRoute() {
  const { isLoggedIn } = useAuth(); // Obtiene el estado de login del "cerebro"

  if (!isLoggedIn) {
    // Si no está logueado, redirige a la página de login
    return <Navigate to="/login" replace />;
  }

  // Si está logueado, muestra la página hija (ej. UploadForm)
  // <Outlet /> es el marcador de posición para las rutas anidadas
  return <Outlet />;
}

export default ProtectedRoute;