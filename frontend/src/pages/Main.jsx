import React, { useState } from 'react';
import Header from '../components/Header.jsx';
import Sidebar from '../components/Sidebar.jsx';
import exo from '../assets/images/exo5.jpg';
import './main.css';

const Main=() => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedPlanet, setSelectedPlanet] = useState({
    name: 'KEPLER-442b',
    type: 'Super Earth',
    distance: '1,206 light years',
    discovered: '2015',
    star: 'Kepler-442',
    mass: '2.36 Earth masses',
    radius: '1.34 Earth radii',
    temperature: '-40°C',
    orbitalPeriod: '112.3 days'
  });

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="main-container">
      <Header onMenuClick={toggleSidebar} />
      
      <div className="hero-section">
        <img 
          src={exo}
          alt="Exoplanet" 
          className="hero-image"
        />
        
        <div className="hero-overlay">
          <div className="hero-content">
            <div className="planet-navigation">
              <button className="nav-planet active">KEPLER-442b</button>
              <button className="nav-planet">TRAPPIST-1e</button>
              <button className="nav-planet">PROXIMA B</button>
            </div>
            
            <div className="hero-title">
              <p className="subtitle">POTENTIALLY HABITABLE EXOPLANET</p>
              <h1 className="title">
                KEPLER<br/>
                442b
              </h1>
              <p className="description">
                The Kepler-442 star has 61% of the Sun's luminosity, with a size of about 97% and a 
                surface temperature of approximately 4,402 K. This planet is one of the most 
                habitable exoplanets in our solar system.
              </p>
            </div>

            <button className="explore-btn" onClick={toggleSidebar}>
              EXPLORE DATA →
            </button>
          </div>
        </div>

        <div className="stats-bottom">
          <div className="stat-item">
            <div className="stat-number">5,000+</div>
            <div className="stat-label">EXOPLANETS DISCOVERED</div>
          </div>
          <div className="stat-divider"></div>
          <div className="stat-item">
            <div className="stat-number">3,800+</div>
            <div className="stat-label">PLANETARY SYSTEMS</div>
          </div>
          <div className="stat-divider"></div>
          <div className="stat-item">
            <div className="stat-number">50+</div>
            <div className="stat-label">POTENTIALLY HABITABLE</div>
          </div>
        </div>
      </div>

      <Sidebar isOpen={sidebarOpen} onClose={toggleSidebar} planetData={selectedPlanet} />
    </div>
  );
}

export default Main;