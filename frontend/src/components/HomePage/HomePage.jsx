// frontend/src/components/HomePage.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import './HomePage.css'; // Crearemos este CSS ahora

function HomePage() {
  const { isLoggedIn } = useAuth();

  // El botón llevará a registrarse si no estás logueado,
  // o directo al análisis si ya lo estás.
  const ctaLink = isLoggedIn ? "/analysis" : "/register";
  const ctaText = isLoggedIn ? "Crear Nuevo Análisis" : "Empezar Ahora";

  return (
    <div className="hero-container">
      {/* Columna Izquierda: Texto */}
      <div className="hero-content">
        <h1 className="hero-headline">
            
          Tu Entrenador Personal
          <span className="hero-headline-highlight"> Impulsado por IA.</span>
        </h1>
        <p className="hero-subheadline">
          Deja de adivinar. Obtén rutinas de gimnasio personalizadas
          basadas en tu físico real y tus objetivos.
          Tu plan evoluciona contigo.
        </p>
        <Link to={ctaLink} className="hero-cta-button">
          {ctaText}
        </Link>
      </div>

      {/* Columna Derecha: Visual (como en la imagen) */}
      <div className="hero-visual">
        {/* Dejamos este espacio para el fondo de partículas.
            Podríamos incluso poner el <ModelViewer /> aquí
            con un modelo 3D genérico si quisiéramos. */}
      </div>
    </div>
  );
}

export default HomePage;