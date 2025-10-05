import { useEffect, useRef } from 'react';
import * as THREE from 'three';

const StellarSystem3D = ({ planetData }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000510);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 15;
    camera.position.y = 5;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    mountRef.current.appendChild(renderer.domElement);

    // Add stars background
    const starsGeometry = new THREE.BufferGeometry();
    const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.1 });
    const starsVertices = [];
    for (let i = 0; i < 5000; i++) {
      const x = (Math.random() - 0.5) * 2000;
      const y = (Math.random() - 0.5) * 2000;
      const z = (Math.random() - 0.5) * 2000;
      starsVertices.push(x, y, z);
    }
    starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);

    // Central Star
    const starRadius = planetData.stellarRadius || 1.0;
    const starGeometry = new THREE.SphereGeometry(starRadius, 32, 32);
    const starMaterial = new THREE.MeshBasicMaterial({ 
      color: getTempColor(planetData.stellarTemp || 5800),
      emissive: getTempColor(planetData.stellarTemp || 5800),
      emissiveIntensity: 0.8
    });
    const star = new THREE.Mesh(starGeometry, starMaterial);
    scene.add(star);

    // Star glow effect
    const glowGeometry = new THREE.SphereGeometry(starRadius * 1.3, 32, 32);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: getTempColor(planetData.stellarTemp || 5800),
      transparent: true,
      opacity: 0.3
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    scene.add(glow);

    // Planet
    const planetRadius = (planetData.planetRadius || 1.0) * 0.3; // Scale down for visibility
    const orbitRadius = 5 + (planetData.orbitalPeriod || 10) * 0.05; // Scale orbit
    
    const planetGeometry = new THREE.SphereGeometry(planetRadius, 32, 32);
    const planetMaterial = new THREE.MeshStandardMaterial({
      color: getPlanetColor(planetData.planetRadius || 1.0, planetData.equilibriumTemp || 300),
      emissive: 0x112244,
      emissiveIntensity: 0.2
    });
    const planet = new THREE.Mesh(planetGeometry, planetMaterial);
    scene.add(planet);

    // Orbit path
    const orbitCurve = new THREE.EllipseCurve(
      0, 0,
      orbitRadius, orbitRadius,
      0, 2 * Math.PI,
      false,
      0
    );
    const orbitPoints = orbitCurve.getPoints(128);
    const orbitGeometry = new THREE.BufferGeometry().setFromPoints(
      orbitPoints.map(p => new THREE.Vector3(p.x, 0, p.y))
    );
    const orbitMaterial = new THREE.LineBasicMaterial({ 
      color: 0x4488ff,
      transparent: true,
      opacity: 0.4
    });
    const orbitLine = new THREE.Line(orbitGeometry, orbitMaterial);
    scene.add(orbitLine);

    // Habitable zone visualization
    const habitableInner = 3;
    const habitableOuter = 7;
    const hzGeometry = new THREE.RingGeometry(habitableInner, habitableOuter, 64);
    const hzMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.1
    });
    const habitableZone = new THREE.Mesh(hzGeometry, hzMaterial);
    habitableZone.rotation.x = Math.PI / 2;
    scene.add(habitableZone);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 2, 100);
    pointLight.position.set(0, 0, 0);
    scene.add(pointLight);

    // Animation
    let angle = 0;
    const orbitalSpeed = 0.01 / (planetData.orbitalPeriod || 10) * 10;

    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      // Rotate star
      star.rotation.y += 0.001;
      glow.rotation.y -= 0.001;

      // Orbit planet
      angle += orbitalSpeed;
      planet.position.x = Math.cos(angle) * orbitRadius;
      planet.position.z = Math.sin(angle) * orbitRadius;
      planet.rotation.y += 0.02;

      // Rotate background stars slowly
      stars.rotation.y += 0.0001;

      // Camera auto-rotation
      camera.position.x = Math.cos(Date.now() * 0.0001) * 15;
      camera.position.z = Math.sin(Date.now() * 0.0001) * 15;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [planetData]);

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden border-2 border-blue-500/30">
      <div ref={mountRef} className="w-full h-full" />
      
      {/* Info Overlay */}
      <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm rounded-lg p-4 text-white text-sm space-y-2 border border-blue-500/30">
        <div className="font-bold text-blue-400 mb-2">System Parameters</div>
        <div className="flex justify-between gap-6">
          <span className="text-gray-400">Star Type:</span>
          <span className="font-mono">{getStarType(planetData.stellarTemp || 5800)}</span>
        </div>
        <div className="flex justify-between gap-6">
          <span className="text-gray-400">Planet Radius:</span>
          <span className="font-mono">{(planetData.planetRadius || 1).toFixed(2)} R⊕</span>
        </div>
        <div className="flex justify-between gap-6">
          <span className="text-gray-400">Orbital Period:</span>
          <span className="font-mono">{(planetData.orbitalPeriod || 10).toFixed(1)} days</span>
        </div>
        <div className="flex justify-between gap-6">
          <span className="text-gray-400">Temperature:</span>
          <span className="font-mono">{(planetData.equilibriumTemp || 300).toFixed(0)} K</span>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur-sm rounded-lg p-3 text-white text-xs space-y-1 border border-blue-500/30">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
          <span>Central Star</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <span>Exoplanet</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500/30 border border-green-500"></div>
          <span>Habitable Zone</span>
        </div>
      </div>
    </div>
  );
};

// Helper functions
function getTempColor(temp) {
  if (temp < 3500) return 0xff4400; // Red dwarf
  if (temp < 5000) return 0xff8800; // Orange
  if (temp < 6000) return 0xffff00; // Yellow
  if (temp < 7500) return 0xffffaa; // White
  return 0xaaaaff; // Blue
}

function getStarType(temp) {
  if (temp < 3500) return 'M-type (Red Dwarf)';
  if (temp < 5000) return 'K-type (Orange)';
  if (temp < 6000) return 'G-type (Sun-like)';
  if (temp < 7500) return 'F-type (Yellow-White)';
  if (temp < 10000) return 'A-type (White)';
  return 'B-type (Blue-White)';
}

function getPlanetColor(radius, temp) {
  if (radius < 1.5) {
    return temp < 273 ? 0x8888ff : temp > 373 ? 0xff8844 : 0x4488ff; // Rocky planet
  } else if (radius < 4) {
    return 0x88ddff; // Neptune-like
  } else {
    return 0xddaa88; // Gas giant
  }
}

export default StellarSystem3D;