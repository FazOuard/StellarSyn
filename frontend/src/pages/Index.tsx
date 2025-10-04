import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";

import Kepler2 from "@/assets/k2.jpg";
import Kepler from "@/assets/kepler3.jpg";
import Tess from "@/assets/tess.jpg";

const Index = () => {
  const navigate = useNavigate();
  const [currentPlanetIndex, setCurrentPlanetIndex] = useState(0);

  const planets = [
    { image: Kepler, name: "Kepler" },
    { image: Kepler2, name: "Kepler2" },
    { image: Tess, name: "TESS" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPlanetIndex((prev) => (prev + 1) % planets.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-screen bg-black relative overflow-hidden">
      {/* Starfield background */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-5" />

      {/* Full background rotating planets */}
      <div className="absolute inset-0">
        {planets.map((planet, index) => (
          <img
            key={index}
            src={planet.image}
            alt={planet.name}
            className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 ${
              index === currentPlanetIndex ? "opacity-100" : "opacity-0"
            }`}
            style={{
              filter: "brightness(0.7) drop-shadow(0 0 100px rgba(59, 130, 246, 0.2))",
            }}
          />
        ))}
      </div>

      {/* Navigation */}
      <nav className="relative z-10 container mx-auto px-8 py-8">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <span className="text-3xl md:text-4xl font-bold tracking-wider text-white">
              StellarSyn
            </span>
          </div>
          <div className="flex gap-8 text-sm text-white/80">
            <button className="hover:text-white transition-colors text-lg">
                DISCOVER
            </button>
            <button
              onClick={() => navigate("/planets")}
              className="hover:text-white transition-colors text-lg"
            >
              PLANETS
            </button>
            <button
              onClick={() => navigate("/classifier")}
              className="hover:text-white transition-colors text-lg"
            >
              RESEARCH
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
        <section className="relative z-20 container mx-auto px-8 flex items-start pt-24 h-[calc(100vh-180px)]">
          <div className="max-w-2xl space-y-8">
            <div className="space-y-4">
              <p className="text-white/60 text-sm tracking-widest uppercase">
                Potentially Habitable Exoplanet
              </p>
              <h1 className="text-7xl font-bold text-white leading-none">
                {planets[currentPlanetIndex].name}
              </h1>
            </div>

            <p className="text-white/70 text-lg leading-relaxed max-w-xl">
              {currentPlanetIndex === 0 && "Kepler-442b is a super-Earth exoplanet orbiting the K-type star Kepler-442, about 1,200 light-years away. It has a radius around 1.3 times that of Earth and lies within its star's habitable zone, making it one of the most Earth-like exoplanets discovered."}
              {currentPlanetIndex === 1 && "The K2 Mission (Kepler's Second Light) continued NASA's search for exoplanets after the original Kepler mission. Repurposed in 2014, K2 observed fields along the ecliptic plane, discovering hundreds of new exoplanets and expanding our understanding of planetary systems."}
              {currentPlanetIndex === 2 && "The Transiting Exoplanet Survey Satellite (TESS) is a NASA mission launched in 2018 to search for exoplanets around nearby bright stars. It has identified thousands of planet candidates, many of which are small, potentially rocky worlds in their stars' habitable zones."}
            </p>
          </div>
        </section>

      {/* Stats Section */}
      <section className="relative z-30 container mx-auto px-8 pb-8 -mt-20">
        {/* <div>
         <Button
            onClick={() => navigate("/classifier")}
            size="lg"
            variant="outline"
            className="border-white/30 text-white hover:bg-white/10 backdrop-blur-sm"
        >
            TRY EXPLORER →
        </Button>
        </div> */}
        <div className="flex gap-20 max-w-4xl">
          <div className="space-y-2">
            <div className="text-5xl font-bold text-white">5,000+</div>
            <div className="text-white/50 text-sm tracking-wider uppercase">
              Exoplanets<br /> Discovered
            </div>
          </div>
          <div className="h-20 w-px bg-white/20" />
          <div className="space-y-2">
            <div className="text-5xl font-bold text-white">3,800+</div>
            <div className="text-white/50 text-sm tracking-wider uppercase">
              Planetary<br /> Systems
            </div>
          </div>
          <div className="h-20 w-px bg-white/20" />
          <div className="space-y-2">
            <div className="text-5xl font-bold text-white">50+</div>
            <div className="text-white/50 text-sm tracking-wider uppercase">
              Potentially<br /> Habitable
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;
