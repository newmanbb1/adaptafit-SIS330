// frontend/src/components/UploadForm.jsx

import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // <-- 1. Importar
import { useAuth } from "../context/AuthContext";

import Stepper from "./Stepper";
import Step1_Info from "./FormSteps/Step1_Info";
import Step2_Goals from "./FormSteps/Step2_Goals";
import Step3_Lifestyle from "./FormSteps/Step3_Lifestyle";
import Step4_Images from "./FormSteps/Step4_Images";

const steps = ["Información", "Objetivos", "Estilo de Vida", "Fotos"];

function UploadForm() {
  const navigate = useNavigate();
  const { token } = useAuth(); // <-- 2. Obtener el usuario autenticado

  const [formData, setFormData] = useState({
    nivel_entrenamiento: "Principiante (No sé qué rutina seguir)",
    frecuencia_entrenamiento: "3 veces",
    objetivo_principal: "Ganancia de masa muscular",
    genero: "Masculino",
    fecha_nacimiento: "",
    altura: "",
    peso_actual: "",
    lesiones: "No",
    lesion_detalle: "",
    horas_sueño: "7-8 horas",
    nivel_estres: "Moderado",
    tipo_trabajo: "Sedentario (oficina)",
  });

  const [frontImage, setFrontImage] = useState(null);
  const [sideImage, setSideImage] = useState(null);
  const [backImage, setBackImage] = useState(null);

  const [currentStep, setCurrentStep] = useState(1);
  const [jobId, setJobId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const nextStep = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setJobId(null);

    if (
      !frontImage ||
      !sideImage ||
      !backImage ||
      !formData.fecha_nacimiento ||
      !formData.altura ||
      !formData.peso_actual
    ) {
      setError(
        "Por favor, completa todos los campos y sube las tres imágenes."
      );
      setIsLoading(false);
      return;
    }

    const submissionData = new FormData();
    for (const key in formData) {
      submissionData.append(key, formData[key]);
    }
    submissionData.append("front_image", frontImage);
    submissionData.append("side_image", sideImage);
    submissionData.append("back_image", backImage);

    try {
      const response = await fetch("http://127.0.0.1:8000/analysis", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          // NO pongas 'Content-Type': 'multipart/form-data',
          // el navegador lo hace automáticamente con FormData
        },
        body: submissionData,
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.statusText}`);
      }

      const result = await response.json();
      //setJobId(result.job_id);
      navigate(`/analysis/${result.job_id}`);
    } catch (err) {
      setError(`Ocurrió un error al enviar: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="form-container wizard-form">
      <Stepper steps={steps} currentStep={currentStep} />

      {error && <p className="message message-error">{error}</p>}

      {/* El <form> ahora envuelve los pasos */}
      <form onSubmit={handleSubmit}>
        {currentStep === 1 && (
          <Step1_Info
            data={formData}
            handleChange={handleInputChange}
            nextStep={nextStep}
          />
        )}

        {currentStep === 2 && (
          <Step2_Goals
            data={formData}
            handleChange={handleInputChange}
            nextStep={nextStep}
            prevStep={prevStep}
          />
        )}

        {currentStep === 3 && (
          <Step3_Lifestyle
            data={formData}
            handleChange={handleInputChange}
            nextStep={nextStep}
            prevStep={prevStep}
          />
        )}

        {currentStep === 4 && (
          <Step4_Images
            // Pasamos los 'setters' de imágenes
            setFrontImage={setFrontImage}
            setSideImage={setSideImage}
            setBackImage={setBackImage}
            prevStep={prevStep}
            isLoading={isLoading}
            // El botón de submit está dentro de este componente
          />
        )}
      </form>
    </div>
  );
}

export default UploadForm;
