// frontend/src/components/AnimatedBackground.jsx
import React, { useCallback } from 'react';
import Particles from 'react-tsparticles';
import { loadSlim } from 'tsparticles-slim'; // Carga el motor ligero
import particlesConfig from '../particles-config'; // Importa nuestra config

function AnimatedBackground() {
  const particlesInit = useCallback(async (engine) => {
    // Carga el motor 'slim'
    await loadSlim(engine);
  }, []);

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      options={particlesConfig}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -1, // Lo pone detrás de todo el contenido
      }}
    />
  );
}

export default AnimatedBackground;