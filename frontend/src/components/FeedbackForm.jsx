// frontend/src/components/FeedbackForm.jsx
import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';

function FeedbackForm({ jobId }) {
  const [feedbackText, setFeedbackText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const { token } = useAuth();
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setSuccess(false);

    if (feedbackText.trim().length < 10) {
      setError('Por favor, escribe un feedback un poco más detallado.');
      setIsLoading(false);
      return;
    }

    // Usamos FormData porque el backend espera 'Form(...)'
    const formData = new FormData();
    formData.append('feedback_text', feedbackText);

    try {
      const response = await fetch(`http://127.0.0.1:8000/feedback/${jobId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('No se pudo enviar el feedback.');
      }

      const result = await response.json();
      console.log('Análisis de feedback:', result.analisis);
      setSuccess(true);
      setFeedbackText(''); // Limpia el formulario
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="feedback-form-container">
      <h2>¿Cómo te sentiste en esta sesión?</h2>
      <p>Tu IA adaptará tu próxima rutina basándose en tu feedback. ¿Hubo algún dolor? ¿Algún ejercicio te gustó?</p>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <textarea
            rows="5"
            placeholder="Ej: La rutina fue increíble, pero las sentadillas me dolieron un poco en la rodilla derecha..."
            value={feedbackText}
            onChange={(e) => setFeedbackText(e.target.value)}
            disabled={isLoading}
          ></textarea>
        </div>
        
        <button type="submit" disabled={isLoading} className="submit-button">
          {isLoading ? 'Analizando Feedback...' : 'Enviar Feedback'}
        </button>
        
        {success && <p className="message message-success">¡Feedback enviado! Tu próxima rutina será ajustada.</p>}
        {error && <p className="message message-error">{error}</p>}
      </form>
    </div>
  );
}

export default FeedbackForm;