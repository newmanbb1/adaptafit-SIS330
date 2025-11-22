// frontend/src/components/Stepper.jsx
import React from 'react';
import './Stepper.css'; // Crearemos este CSS en el Paso 4

function Stepper({ steps, currentStep }) {
  return (
    <div className="stepper-wrapper">
      {steps.map((step, index) => {
        const stepNumber = index + 1;
        let stepClass = "stepper-item";
        if (stepNumber === currentStep) {
          stepClass += " active";
        } else if (stepNumber < currentStep) {
          stepClass += " completed";
        }

        return (
          <div className={stepClass} key={step}>
            <div className="step-counter">{stepNumber}</div>
            <div className="step-name">{step}</div>
          </div>
        );
      })}
    </div>
  );
}

export default Stepper;