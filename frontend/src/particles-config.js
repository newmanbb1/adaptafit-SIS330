// frontend/src/particles-config.js

const particlesConfig = {
  background: {
    color: {
      value: "#121212", // Coincide con tu --main-bg
    },
  },
  fpsLimit: 60,
  interactivity: {
    events: {
      onClick: {
        enable: true,
        mode: "push",
      },
      onHover: {
        enable: true,
        mode: "grab",
      },
    },
    modes: {
      push: {
        quantity: 4,
      },
      grab: {
        distance: 200,
        links: {
          opacity: 0.5,
        },
      },
    },
  },
  particles: {
    color: {
      value: ["#FFFFFF", "#007FFF"], // Blanco y Azul Eléctrico
    },
    links: {
      color: "#FFFFFF",
      distance: 150,
      enable: true,
      opacity: 0.2, // Líneas de la red sutiles
      width: 1,
    },
    move: {
      direction: "none",
      enable: true,
      outModes: {
        default: "out",
      },
      random: false,
      speed: 0.5, // Movimiento lento
      straight: false,
    },
    number: {
      density: {
        enable: true,
      },
      value: 100, // Cantidad de partículas
    },
    opacity: {
      value: { min: 0.1, max: 0.5 },
    },
    shape: {
      type: "circle",
    },
    size: {
      value: { min: 1, max: 3 },
    },
  },
  detectRetina: true,
};

export default particlesConfig;