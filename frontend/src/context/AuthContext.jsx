// frontend/src/context/AuthContext.jsx
import React, { createContext, useState, useContext } from 'react';
import { jwtDecode } from 'jwt-decode';
// 1. Crear el Contexto
const AuthContext = createContext();

// 2. Crear el Proveedor (el componente que "envuelve" la app)
export function AuthProvider({ children }) {
  // Usamos localStorage para recordar el token entre recargas de página
  const [token, setToken] = useState(localStorage.getItem('token'));
const [userEmail, setUserEmail] = useState(localStorage.getItem('userEmail'));
  const login = (newToken) => {
    const decodedToken = jwtDecode(newToken);
    const email = decodedToken.sub;
    localStorage.setItem('token', newToken);
    localStorage.setItem('userEmail', email);
    setToken(newToken);
    setUserEmail(email);
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userEmail');
    setToken(null);
    setUserEmail(null);
  };

  // El valor que compartiremos con toda la app
  const value = {
    token,
    userEmail,
    isLoggedIn: !!token, // Es 'true' si 'token' no es null
    login,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// 3. Crear un "hook" personalizado para usar el contexto fácilmente
export function useAuth() {
  return useContext(AuthContext);
}