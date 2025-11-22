import React from 'react';

function Step1_Info({ data, handleChange, nextStep }) {
  return (
    <div className="form-step">
      <h3>Información Básica</h3>
      <div className="form-group">
        <label>¿Cuál es tu género?</label>
        <select name="genero" value={data.genero} onChange={handleChange}>
          <option>Masculino</option> <option>Femenino</option>
        </select>
      </div>
      <div className="form-group">
        <label>Fecha de Nacimiento</label>
        <input type="date" name="fecha_nacimiento" value={data.fecha_nacimiento} onChange={handleChange} />
      </div>
      <div className="form-group">
        <label>Altura (cm)</label>
        <input type="number" name="altura" placeholder="Ej: 175" value={data.altura} onChange={handleChange} />
      </div>
      <div className="form-group">
        <label>Peso Actual (kg)</label>
        <input type="number" step="0.1" name="peso_actual" placeholder="Ej: 70.5" value={data.peso_actual} onChange={handleChange} />
      </div>
      <div className="form-navigation">
        <button type="button" onClick={nextStep} className="submit-button">Siguiente</button>
      </div>
    </div>
  );
}
export default Step1_Info;