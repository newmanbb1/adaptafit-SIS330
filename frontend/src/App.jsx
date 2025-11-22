// frontend/src/App.jsx

import React from "react";
import {
  Routes,
  Route,
  useNavigate,
  Link,
  useLocation,
} from "react-router-dom"; // <-- 1. Importar
import { useAuth } from "./context/AuthContext";

import UploadForm from "./components/UploadForm";
import LoadingScreen from "./components/LoadingScreen"; // <-- 2. Importar nuevos
import ResultsDashboard from "./components/ResultsDashboard/ResultsDashboard"; // <-- 2. Importar nuevos

import LoginPage from "./components/LoginPage";
import RegisterPage from "./components/RegisterPage";
import ProtectedRoute from "./components/ProtectedRoute"; // <-- 2. Importa el guardián
import HistoryPage from "./components/HistoryPage";

import AnimatedBackground from "./components/AnimatedBackground";
import logo from './assets/logo_black.png'; 
import "./App.css";
import HomePage from './components/HomePage/HomePage';

function App() {
  const { isLoggedIn, logout } = useAuth(); // <-- 3. Obtén el estado y la función de logout
  const navigate = useNavigate();
  const location = useLocation();

// --- 1. Lógica de Layout (Actualizada) ---
  const isAuthPage = location.pathname === '/login' || location.pathname === '/register';
  const isHomePage = location.pathname === '/';

const showAnimatedBg = isAuthPage || isHomePage;

  const showNavMenu = !isAuthPage && !isHomePage;
  const layoutClass = showNavMenu ? "app-layout-dashboard" : "app-layout-full";
  const handleLogout = () => {
    logout();
    navigate("/login"); // Redirige a login después de cerrar sesión
  };

  return (
    <div className="app-layout">
      {showAnimatedBg && <AnimatedBackground />}
      {showNavMenu && (
        <nav className="nav-menu">
          <div className="logo-container">
            <Link to="/">
             <img src={logo} alt="Logo de la App" className="nav-logo" />
            </Link>
          </div>
          <ul>
            {isLoggedIn ? (
              <>
                <li>
                  <Link to="/analysis">Análisis</Link>
                </li>
                <li>
                  <Link to="/history">Historial</Link>
                </li>
                <li>
                  <button onClick={handleLogout} className="logout-button">
                    Cerrar Sesión
                  </button>
                </li>
              </>
            ) : (
              <>
                <li>
                  <Link to="/login">Iniciar Sesión</Link>
                </li>
                <li>
                  <Link to="/register">Registrarse</Link>
                </li>
              </>
            )}
          </ul>
        </nav>
      )}

      
      

{showAnimatedBg ? (
        // --- ESTE ES EL NUEVO WRAPPER PARA LOGIN/REGISTER ---
        <div className="auth-pages-layout">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
          </Routes>
        </div>
      ) : (
        <main className="main-content">
          <Routes>
            {/* --- Rutas Privadas (Protegidas) --- */}
            <Route element={<ProtectedRoute />}>
              <Route path="/analysis" element={<UploadForm />} />
              <Route path="/analysis/:jobId" element={<LoadingScreen />} />
              <Route path="/results/:jobId" element={<ResultsDashboard />} />
              <Route path="/history" element={<HistoryPage />} />
            </Route>
          </Routes>
        </main>
      )}
    </div>
  );
}
export default App;
