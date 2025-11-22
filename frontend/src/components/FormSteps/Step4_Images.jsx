import React from 'react';

function Step4_Images({ setFrontImage, setSideImage, setBackImage, prevStep, isLoading }) {
  return (
    <div className="form-step">
      <h3>Sube tus Fotos</h3>
      <p style={{color: 'var(--text-secondary)'}}>Para el análisis 3D, necesitamos tres fotos: frontal, de perfil y de espalda. Asegúrate de tener buena iluminación.</p>
      <div className="form-group">
        <label>Foto Frontal:</label>
        <input type="file" name="front_image" accept="image/*" onChange={(e) => setFrontImage(e.target.files[0])} />
      </div>
      <div className="form-group">
        <label>Foto de Perfil:</label>
        <input type="file" name="side_image" accept="image/*" onChange={(e) => setSideImage(e.target.files[0])} />
      </div>
      <div className="form-group">
        <label>Foto de Espalda:</label>
        <input type="file" name="back_image" accept="image/*" onChange={(e) => setBackImage(e.target.files[0])} />
      </div>
      <div className="form-navigation button-group">
        <button type="button" onClick={prevStep} className="button-secondary">Anterior</button>
        {/* Este es el botón de submit final */}
        <button type="submit" disabled={isLoading} className="submit-button">
          {isLoading ? "Analizando..." : "Analizar Mi Cuerpo"}
        </button>
      </div>
    </div>
  );
}
export default Step4_Images;