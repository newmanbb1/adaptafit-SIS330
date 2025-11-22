import React from 'react';

function Step2_Goals({ data, handleChange, nextStep, prevStep }) {
  return (
    <div className="form-step">
      <h3>Objetivos y Experiencia</h3>
      <div className="form-group">
        <label>¿Cuál es tu nivel de entrenamiento?</label>
        <select name="nivel_entrenamiento" value={data.nivel_entrenamiento} onChange={handleChange}>
          <option>Introductorio (no tiene experiencia)</option>
          <option>Principiante (No sé qué rutina seguir)</option>
          <option>Intermedio (Tiene propia rutina de ejercicios)</option>
          <option>Avanzado (Experto suficiente para ser entrenador)</option>
          <option>Experto (Experimentado para ser atleta)</option>
        </select>
      </div>
      <div className="form-group">
        <label>¿Cuál es tu principal objetivo?</label>
        <select name="objetivo_principal" value={data.objetivo_principal} onChange={handleChange}>
          <option>Pérdida de grasa</option>
          <option>Ganancia de masa muscular</option>
          <option>Tonificación</option>
        </select>
      </div>
      <div className="form-group">
        <label>¿Tienes lesiones?</label>
        <select name="lesiones" value={data.lesiones} onChange={handleChange}>
          <option>No</option> <option>Sí</option>
        </select>
        {data.lesiones === "Sí" && (
          <input type="text" name="lesion_detalle" placeholder="Indica la zona lesionada" value={data.lesion_detalle} onChange={handleChange} style={{ marginTop: "10px" }} />
        )}
      </div>
      <div className="form-navigation button-group">
        <button type="button" onClick={prevStep} className="button-secondary">Anterior</button>
        <button type="button" onClick={nextStep} className="submit-button">Siguiente</button>
      </div>
    </div>
  );
}
export default Step2_Goals;