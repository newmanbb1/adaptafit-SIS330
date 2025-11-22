import React from 'react';

function Step3_Lifestyle({ data, handleChange, nextStep, prevStep }) {
  return (
    <div className="form-step">
      <h3>Estilo de Vida</h3>
      <div className="form-group">
        <label>¿Cuántos días a la semana planea entrenar?</label>
        <select name="frecuencia_entrenamiento" value={data.frecuencia_entrenamiento} onChange={handleChange}>
          <option>6 veces</option> <option>5 veces</option> <option>4 veces</option>
          <option>3 veces</option> <option>2 veces</option> <option>1 vez</option>
        </select>
      </div>
      <div className="form-group">
        <label>¿Cuántas horas duermes en promedio?</label>
        <select name="horas_sueño" value={data.horas_sueño} onChange={handleChange}>
          <option>Menos de 6 horas</option> <option>7-8 horas</option> <option>Más de 8 horas</option>
        </select>
      </div>
      <div className="form-group">
        <label>¿Cuál es tu nivel de estrés diario?</label>
        <select name="nivel_estres" value={data.nivel_estres} onChange={handleChange}>
          <option>Bajo</option> <option>Moderado</option> <option>Alto</option>
        </select>
      </div>
      <div className="form-group">
        <label>¿Cómo describirías tu trabajo/actividad diaria?</label>
        <select name="tipo_trabajo" value={data.tipo_trabajo} onChange={handleChange}>
          <option>Sedentario (oficina)</option> <option>Activo (movimiento ligero)</option> <option>Físico (trabajo pesado)</option>
        </select>
      </div>
      <div className="form-navigation button-group">
        <button type="button" onClick={prevStep} className="button-secondary">Anterior</button>
        <button type="button" onClick={nextStep} className="submit-button">Siguiente</button>
      </div>
    </div>
  );
}
export default Step3_Lifestyle;