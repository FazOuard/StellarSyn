import React from 'react';
import './Header.css';

function Header({ onMenuClick }) {
  return (
    <header className="header">
      <div className="logo">
        
        <span className="logo-text">EXOPLANET EXPLORER</span>
      </div>
      
      <nav className="nav-menu">
        <a href="#" className="nav-link">DISCOVER</a>
        <a href="#" className="nav-link">PLANETS</a>
        <a href="#" className="nav-link">MISSIONS</a>
        <a href="#" className="nav-link">RESEARCH</a>
      </nav>

      
    </header>
  );
}

export default Header;