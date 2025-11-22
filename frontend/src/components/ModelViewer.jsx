// frontend/src/components/ModelViewer.jsx
import React, { Suspense, useRef, useState, useEffect } from 'react'; // <-- 1. Importamos hooks
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Loader } from '@react-three/drei';

function Model({ modelPath }) {
  const { scene } = useGLTF(modelPath);
  return <primitive object={scene} scale={1.8} />;
}

export default function ModelViewer({ modelUrl }) {
  // 2. Referencia al contenedor del Canvas
  const containerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // 3. Función para alternar pantalla completa
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => {
        console.error(`Error al intentar activar pantalla completa: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  // 4. Escuchar cambios en el sistema (por si el usuario usa "Esc" para salir)
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  return (
    // Asignamos la ref al div contenedor
    <div 
      ref={containerRef} 
      style={{ 
        height: isFullscreen ? '100vh' : '400px', // Ajuste dinámico si es necesario
        width: '100%', 
        background: '#111',
        position: 'relative' // Necesario para posicionar el botón absoluto
      }}
    >
      {/* Botón de Pantalla Completa */}
      <button
        onClick={toggleFullscreen}
        style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          zIndex: 10,
          background: 'rgba(0, 0, 0, 0.6)',
          color: 'white',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          borderRadius: '8px',
          padding: '8px 12px',
          cursor: 'pointer',
          fontSize: '14px',
          fontWeight: 'bold',
          transition: 'all 0.2s ease'
        }}
        onMouseEnter={(e) => e.target.style.background = 'rgba(0, 0, 0, 0.9)'}
        onMouseLeave={(e) => e.target.style.background = 'rgba(0, 0, 0, 0.6)'}
      >
        {isFullscreen ? '⤓ Salir' : '⤢ Pantalla Completa'}
      </button>

      <Canvas camera={{ position: [0, 0, 2], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <Suspense fallback={null}>
          <Model modelPath={modelUrl} />
        </Suspense>
        <OrbitControls />
      </Canvas>
      <Loader />
    </div>
  );
}