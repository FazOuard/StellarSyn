import React, { useState } from 'react';
import Header from '../components/Header.jsx';
import Sidebar from '../components/Sidebar.jsx';
import exo from '../assets/images/exo5.jpg';
import './main.css';

const Main = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const planetsData = {
    kepler: {
      name: 'KEPLER-442b',
      type: 'Super Earth',
      distance: '1,206 light years',
      discovered: '2015',
      star: 'Kepler-442',
      mass: '2.36 Earth masses',
      radius: '1.34 Earth radii',
      temperature: '-40°C',
      orbitalPeriod: '112.3 days',
      title: 'KEPLER',
      subtitle: '442b',
      description: 'The Kepler-442 star has 61% of the Sun\'s luminosity, with a size of about 97% and a surface temperature of approximately 4,402 K. This planet is one of the most habitable exoplanets in our solar system.',
      image: exo
    },
    trappist: {
      name: 'TRAPPIST-1e',
      type: 'Earth-sized',
      distance: '40 light years',
      discovered: '2017',
      star: 'TRAPPIST-1',
      mass: '0.62 Earth masses',
      radius: '0.92 Earth radii',
      temperature: '-22°C',
      orbitalPeriod: '6.1 days',
      title: 'TRAPPIST',
      subtitle: '1e',
      description: 'TRAPPIST-1e orbits an ultra-cool dwarf star and is located in the habitable zone. It receives similar amounts of energy from its star as Earth does from the Sun, making it a prime candidate for habitability studies.',
      image: 'https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4?w=1200&h=800&fit=crop'
    },
    proxima: {
      name: 'PROXIMA B',
      type: 'Terrestrial',
      distance: '4.24 light years',
      discovered: '2016',
      star: 'Proxima Centauri',
      mass: '1.17 Earth masses',
      radius: '1.07 Earth radii',
      temperature: '-39°C',
      orbitalPeriod: '11.2 days',
      title: 'PROXIMA',
      subtitle: 'B',
      description: 'Proxima B orbits the closest star to our Sun, Proxima Centauri. Despite its proximity, the planet faces challenges including stellar flares from its red dwarf host star, but remains one of our best prospects for finding life.',
      image: 'https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=1200&h=800&fit=crop'
    }
  };

  const [activePlanet, setActivePlanet] = useState('kepler');
  const [selectedPlanet, setSelectedPlanet] = useState(planetsData.kepler);
  
  const handlePlanetClick = (planetKey) => {
    setActivePlanet(planetKey);
    setSelectedPlanet(planetsData[planetKey]);
  };
  
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="main-container">
      <Header onMenuClick={toggleSidebar} />
     
      <div className="hero-section">
        <img
          src={selectedPlanet.image}
          alt="Exoplanet"
          className="hero-image"
        />
       
        <div className="hero-overlay">
          <div className="hero-content">
            <div className="planet-navigation">
              <button 
                className={`nav-planet ${activePlanet === 'kepler' ? 'active' : ''}`}
                onClick={() => handlePlanetClick('kepler')}
              >
                KEPLER-442b
              </button>
              <button 
                className={`nav-planet ${activePlanet === 'trappist' ? 'active' : ''}`}
                onClick={() => handlePlanetClick('trappist')}
              >
                TRAPPIST-1e
              </button>
              <button 
                className={`nav-planet ${activePlanet === 'proxima' ? 'active' : ''}`}
                onClick={() => handlePlanetClick('proxima')}
              >
                PROXIMA B
              </button>
            </div>
           
            <div className="hero-title">
              <p className="subtitle">POTENTIALLY HABITABLE EXOPLANET</p>
              <h1 className="title">
                {selectedPlanet.title}<br/>
                {selectedPlanet.subtitle}
              </h1>
              <p className="description">
                {selectedPlanet.description}
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