import React from 'react';
import './sidebar.css';

const Sidebar =({ isOpen, onClose, planetData })=> {
  return (
    <>
      <div className={`sidebar-overlay ${isOpen ? 'active' : ''}`} onClick={onClose}></div>
      <div className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>TECHNICAL DATA</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="sidebar-content">
          <div className="planet-visual">
            <img 
              src="https://images.unsplash.com/photo-1614732414444-096e5f1122d5?w=400&h=400&fit=crop" 
              alt="Planet visualization" 
              className="planet-image"
            />
          </div>

          <div className="data-section">
            <h3>OVERVIEW</h3>
            <div className="data-grid">
              <div className="data-item">
                <span className="data-label">Planet Name</span>
                <span className="data-value">{planetData.name}</span>
              </div>
              <div className="data-item">
                <span className="data-label">Type</span>
                <span className="data-value">{planetData.type}</span>
              </div>
              <div className="data-item">
                <span className="data-label">Distance from Earth</span>
                <span className="data-value">{planetData.distance}</span>
              </div>
              <div className="data-item">
                <span className="data-label">Discovered</span>
                <span className="data-value">{planetData.discovered}</span>
              </div>
            </div>
          </div>

          <div className="data-section">
            <h3>PHYSICAL CHARACTERISTICS</h3>
            <div className="data-grid">
              <div className="data-item">
                <span className="data-label">Mass</span>
                <span className="data-value">{planetData.mass}</span>
              </div>
              <div className="data-item">
                <span className="data-label">Radius</span>
                <span className="data-value">{planetData.radius}</span>
              </div>
              <div className="data-item">
                <span className="data-label">Temperature</span>
                <span className="data-value">{planetData.temperature}</span>
              </div>
              <div className="data-item">
                <span className="data-label">Orbital Period</span>
                <span className="data-value">{planetData.orbitalPeriod}</span>
              </div>
            </div>
          </div>

          <div className="data-section">
            <h3>HOST STAR</h3>
            <div className="data-grid">
              <div className="data-item">
                <span className="data-label">Star Name</span>
                <span className="data-value">{planetData.star}</span>
              </div>
              <div className="data-item">
                <span className="data-label">Spectral Type</span>
                <span className="data-value">K-type main sequence</span>
              </div>
            </div>
          </div>

          <div className="habitability-score">
            <h3>HABITABILITY INDEX</h3>
            <div className="progress-bar">
              <div className="progress-fill" style={{width: '78%'}}></div>
            </div>
            <p className="progress-label">78% Similar to Earth</p>
          </div>
        </div>
      </div>
    </>
  );
}

export default Sidebar;