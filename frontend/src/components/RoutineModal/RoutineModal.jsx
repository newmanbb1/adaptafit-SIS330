// frontend/src/components/RoutineModal.jsx
import React from "react";
import "./RoutineModal.css";

// ❗️❗️ PASO 1: AÑADE TU API KEY AQUÍ
// (Debe ser la misma clave que usaste en el backend)

function RoutineModal({ dia, onClose }) {
  const handleBackdropClick = (e) => {
    if (e.target.className === "modal-backdrop") {
      onClose();
    }
  };

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        <button className="modal-close-button" onClick={onClose}>
          &times;
        </button>

        <div className="modal-header">
          <span>DÍA {dia.dia}</span>
          <h2>{dia.titulo}</h2>
        </div>

        <div className="modal-body">
          <table className="modal-ejercicios-table">
            <thead>
              <tr>
                <th>Ejercicio</th>
                <th>Series</th>
                <th>Reps</th>
                <th>Descanso</th>
                <th>Justificación</th>
                <th>Visual</th>
              </tr>
            </thead>
            <tbody>
              {dia.ejercicios.map((ejercicio, index) => {
                // ❗️❗️ PASO 2: CORRECCIÓN APLICADA
                // 1. Usamos 'ejercicio.gifUrl' (camelCase)
                // 2. Le añadimos la API key a la URL
                const gifSrc = ejercicio.exerciseId
                  ? `http://127.0.0.1:8000/exercise_gif/${ejercicio.exerciseId}`
                  : null;

                return (
                  <tr key={index}>
                    <td data-label="Ejercicio">
                      <strong>{ejercicio.nombre}</strong>
                      <p className="ejercicio-tecnica">
                        <strong>Técnica:</strong> {ejercicio.tecnica}
                      </p>
                    </td>
                    <td data-label="Series">{ejercicio.series}</td>
                    <td data-label="Reps">{ejercicio.repeticiones}</td>
                    <td data-label="Descanso">{ejercicio.descanso}</td>
                    <td
                      data-label="Justificación"
                      className="ejercicio-justificacion"
                    >
                      {ejercicio.justificacion}
                    </td>

                    {/* ❗️❗️ PASO 3: USAMOS LA NUEVA URL CORREGIDA */}
                    <td data-label="Visual">
                      {/* ❗️❗️ PASO 2: USAMOS LA NUEVA URL */}
                      {gifSrc ? (
                        <img
                          src={gifSrc}
                          alt={`Demostración de ${ejercicio.nombre}`}
                          className="exercise-gif"
                          onError={(e) => {
                            e.target.src =
                              "https://via.placeholder.com/100/AAAAAA/FFFFFF?text=GIF+Error";
                          }}
                        />
                      ) : (
                        <p className="no-gif-message">N/A</p>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default RoutineModal;
