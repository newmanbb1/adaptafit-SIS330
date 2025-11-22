// frontend/src/components/ResultsDashboard.jsx
import React, { useState, useEffect, Suspense, useMemo } from "react"; // <-- 1. Añade useMemo
import { useParams, useNavigate } from "react-router-dom";
import ModelViewer from "../ModelViewer";
import FeedbackForm from "../FeedbackForm";
import { useAuth } from "../../context/AuthContext";
import "./ResultsDashboard.css";
import RoutineModal from "../RoutineModal/RoutineModal"; // <-- 2. Importa el nuevo Modal
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

function ResultsDashboard() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { token, userEmail } = useAuth();

  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isRegenerating, setIsRegenerating] = useState(false);

  // --- 3. NUEVO ESTADO para controlar el modal ---
  const [selectedDay, setSelectedDay] = useState(null);

  const LoadingFallback = () => (
    <div className="loading-container">
      <div className="spinner"></div>
      <p>Cargando...</p>
    </div>
  );

  const generatePDF = (rutina) => {
    if (!rutina) return;

    // 1. Inicializa el documento PDF
    const doc = new jsPDF();

    // 2. Añade el Título
    doc.setFontSize(20);
    doc.text("Tu Rutina Semanal Personalizada", 14, 22);

    // 3. Añade el Resumen de la IA (con saltos de línea automáticos)
    doc.setFontSize(11);
    const resumenLines = doc.splitTextToSize(rutina.resumen_ia, 180); // 180 = ancho
    doc.text(resumenLines, 14, 35);

    // 4. Prepara las tablas de ejercicios
    const tableHeaders = [
      ["Ejercicio", "Series", "Reps", "Descanso", "Técnica"],
    ];

    // Posición Y inicial para la primera tabla
    let startY = 35 + resumenLines.length * 5 + 10; // 10 de padding

    rutina.plan_semanal.forEach((dia) => {
      // Si la tabla se va a salir de la página, añade una nueva
      if (startY > 250) {
        doc.addPage();
        startY = 20; // Resetea la posición Y en la nueva página
      }

      // Título del Día
      doc.setFontSize(16);
      doc.text(`Día ${dia.dia}: ${dia.titulo}`, 14, startY);
      startY += 10;

      // Mapea los datos de los ejercicios para la tabla
      const tableBody = dia.ejercicios.map((ej) => [
        ej.nombre,
        ej.series,
        ej.repeticiones,
        ej.descanso,
        ej.tecnica, // Usamos técnica en lugar de justificación (más útil)
      ]);

      // 5. Dibuja la tabla
      autoTable(doc, {
        // <-- Llama a 'autoTable' y pasa 'doc'
        startY: startY,
        head: tableHeaders,
        body: tableBody,
        theme: "striped",
        styles: { fontSize: 8 },
        headStyles: { fillColor: [30, 30, 30] },
      });

      // Actualiza la posición Y para la siguiente tabla
      startY = doc.lastAutoTable.finalY + 15; // 15 de padding
    });

    // 6. Guarda el archivo
    doc.save(`mi-rutina-${jobId.substring(0, 6)}.pdf`);
  };

  const modelUrl = `http://127.0.0.1:8000/jobs/${jobId}/meshy_model.glb`;

  // --- LÓGICA DE FETCH (sin cambios) ---
  useEffect(() => {
    const fetchResults = async () => {
      if (!token) {
        setIsLoading(false);
        return;
      }
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/analysis/status/${jobId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!response.ok)
          throw new Error("No se pudieron cargar los resultados.");
        const data = await response.json();
        if (data.status === "completed" && data.results_data) {
          setResults(data.results_data);
        } else if (data.status === "pending") {
          navigate(`/analysis/${jobId}`);
        } else {
          throw new Error(data.results_data?.error || "El análisis falló.");
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    if (!results) {
      fetchResults();
    }
  }, [jobId, results, token, navigate]);

  // --- LÓGICA DE REGENERAR (sin cambios) ---
  const handleRegenerate = async () => {
    // ... (Tu lógica de handleRegenerate) ...
    setIsRegenerating(true);
    setError(null);
    const formData = new FormData();
    formData.append("regenerate_job_id", jobId);
    try {
      const response = await fetch("http://127.0.0.1:8000/analysis", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!response.ok) throw new Error("Falló la solicitud de regeneración.");
      const result = await response.json();
      navigate(`/analysis/${result.job_id}`);
    } catch (err) {
      setError(err.message);
      setIsRegenerating(false);
    }
  };
  console.log("Results data:", results);
  // --- 4. NUEVA LÓGICA para parsear el JSON de la rutina ---
  // --- 4. NUEVA LÓGICA (CORREGIDA) ---
  const rutinaData = useMemo(() => {
    // 1. Verifica que 'results' exista y que 'rutina_generada' sea un objeto
    if (
      results &&
      results.rutina_generada &&
      typeof results.rutina_generada === "object"
    ) {
      // 2. Verifica que el objeto tenga la estructura que esperamos
      if (
        results.rutina_generada.plan_semanal &&
        Array.isArray(results.rutina_generada.plan_semanal)
      ) {
        // 3. ¡Si es válido, devuélvelo tal cual! (SIN PARSEAR)
        return results.rutina_generada;
      }
    }

    // 4. Si 'results' existe pero la rutina no es válida, muestra el error
    if (results) {
      console.error(
        "Error: 'rutina_generada' no es un objeto válido o no existe."
      );
      return {
        resumen_ia: "Error: El formato de la rutina generada era inválido.",
        plan_semanal: [],
      };
    }

    // 5. Si 'results' aún no carga, devuelve null
    return null;
  }, [results]);

  // --- LÓGICA DE RENDERIZADO (MODIFICADA) ---
  if (isLoading) {
    return <LoadingFallback />;
  }
  if (error) {
    return <p className="message message-error">{error}</p>;
  }
  // Si 'results' o 'rutinaData' aún no están listos
  if (!results || !rutinaData) {
    return <LoadingFallback />;
  }

  const { reporte_3d, somatotipo, arquetipo } = results;
  const { resumen_ia, plan_semanal } = rutinaData;

  // --- 5. ESTE ES EL NUEVO RENDERIZADO ---
  return (
    <>
      {" "}
      {/* Usamos un Fragment para que el Modal pueda estar fuera del layout */}
      <div className="results-container">
        {/* --- Encabezado y Botón de Regenerar (Sin cambios) --- */}
        <div className="results-header">
          <h1>Tu Análisis y Rutina : {userEmail}</h1>
          <div className="header-actions">
            <p>¿Quieres tu próxima rutina basada en tu feedback?</p>
            <div className="action-buttons">
              <button
                onClick={handleRegenerate}
                disabled={isRegenerating}
                className="submit-button"
              >
                {isRegenerating
                  ? "Regenerando..."
                  : "Nueva rutina con feedback"}
              </button>

              <button
                onClick={() => generatePDF(rutinaData)}
                className="submit-button secondary-button"
              >
                Descargar PDF
              </button>
            </div>
          </div>
        </div>

        {/* --- Layout de Grid Principal (Sin cambios) --- */}
        <div className="results-grid-layout">
          {/* --- Columna Principal (MODIFICADA PARA CARDS) --- */}
          <main className="results-main-col">
            <h2>Tu Plan Semanal</h2>

            {/* Resumen de la IA */}
            <p className="ai-summary">{resumen_ia}</p>

            {/* --- AQUÍ VAN LAS "CARDS" --- */}
            <div className="day-cards-grid">
              {plan_semanal.length > 0 ? (
                plan_semanal.map((dia) => (
                  <div
                    key={dia.dia}
                    className="day-card"
                    onClick={() => setSelectedDay(dia)} // <-- Abre el modal
                  >
                    <span>DÍA {dia.dia}</span>
                    <h3>{dia.titulo}</h3>
                  </div>
                ))
              ) : (
                <p>No se pudo generar un plan de entrenamiento.</p>
              )}
            </div>
          </main>

          {/* --- Columna Lateral (Sin cambios) --- */}
          <aside className="results-sidebar-col">
            <div className="model-viewer-container card-box">
              <h3>Tu Modelo 3D</h3>
              <p>Puedes rotar, mover y hacer zoom.</p>
              <Suspense fallback={<LoadingFallback />}>
                <ModelViewer modelUrl={modelUrl} />
              </Suspense>
            </div>

            <div className="analysis-summary card-box">
              <h3>Resumen de Análisis</h3>
              <ul className="summary-list">
                <li>
                  {/* Somatotipo */}
                  <strong>Complexión Física:</strong>
                  <span>
                    {/* Usamos el dato {somatotipo} para personalizar el texto si es posible */}
                    {somatotipo === "Mesomorfo"
                      ? "Tienes una complexión atlética natural (Mesomorfo). Tu cuerpo responde bien al entrenamiento."
                      : somatotipo || "N/A"}
                  </span>
                </li>

                <li>
                  {/* Arquetipo */}
                  <strong>Estilo de Vida:</strong>
                  <span>
                    {arquetipo === "Estilo de Vida Equilibrado"
                      ? "Llevas un estilo de vida equilibrado. Un gran punto de partida para tus objetivos."
                      : arquetipo || "N/A"}
                  </span>
                </li>

                <li>
                  {/* Altura */}
                  <strong>Altura:</strong>
                  <span>{reporte_3d?.altura_cm} cm</span>
                </li>

                <li>
                  {/* Volumen */}
                  <strong>Volumen Corporal:</strong>
                  <span>
                    {/* El número principal */}
                    <strong>{reporte_3d?.volumen_total_litros} L</strong>
                    {/* La explicación */}
                    <small>
                      Mide el espacio total que ocupa tu cuerpo. Es un dato
                      avanzado que la IA usa (junto con tu peso) para estimar tu
                      composición corporal general.
                    </small>
                  </span>
                </li>

                <li>
                  {/* ICC */}
                  <strong>Distribución de Grasa (Cintura-Cadera):</strong>
                  <span>
                    {reporte_3d?.indices_corporales?.indice_cintura_cadera}
                  </span>
                </li>

                <li>
                  {/* ICA */}
                  <strong>Proporción Corporal (Cintura-Altura):</strong>
                  <span>
                    {reporte_3d?.indices_corporales?.indice_cintura_altura} 
                  </span>
                </li>
              </ul>
            </div>
          </aside>
        </div>

        {/* --- Sección de Feedback (Sin cambios) --- */}
        <hr style={{ borderColor: "var(--border-color)", margin: "40px 0" }} />
        <div className="card-box">
          <FeedbackForm jobId={jobId} />
        </div>
      </div>
      {/* --- 6. EL MODAL (se renderiza fuera del layout) --- */}
      {selectedDay && (
        <RoutineModal dia={selectedDay} onClose={() => setSelectedDay(null)} />
      )}
    </>
  );
}

export default ResultsDashboard;
