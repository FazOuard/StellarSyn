import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from "@/components/ui/carousel";
import planet1 from "@/assets/planet-1.jpg";
import planet2 from "@/assets/planet-2.jpg";
import planet3 from "@/assets/planet-3.jpg";
import planet4 from "@/assets/planet-4.jpg";

const Planets = () => {
  const navigate = useNavigate();

  const planetsData = [
    {
      image: planet1,
      name: "KEPLER-442b",
      type: "Super-Earth",
      distance: "1,206 light-years",
      discoveryYear: "2015",
      mass: "2.36 Earth masses",
      radius: "1.34 Earth radii",
      orbitalPeriod: "112.3 days",
      temperature: "233 K (-40°C)",
      description: "The Kepler-442 star has 61% of the Sun's luminosity, with a size of about 97% and a surface temperature of approximately 4,402 K. This planet is one of the most habitable exoplanets discovered, orbiting within the habitable zone where liquid water could exist on its surface. It receives about 70% of the light Earth gets from the Sun.",
      habitability: "0.836 (Very High)",
      composition: "Rocky with possible water content"
    },
    {
      image: planet2,
      name: "TRAPPIST-1e",
      type: "Rocky Planet",
      distance: "40 light-years",
      discoveryYear: "2017",
      mass: "0.62 Earth masses",
      radius: "0.92 Earth radii",
      orbitalPeriod: "6.1 days",
      temperature: "246 K (-27°C)",
      description: "Located 40 light-years away in the constellation Aquarius, TRAPPIST-1e orbits within its star's habitable zone. With similar size to Earth, it may harbor liquid water on its surface. Part of a remarkable system of seven Earth-sized planets, TRAPPIST-1e is considered one of the best candidates for potential habitability.",
      habitability: "0.859 (Very High)",
      composition: "Rocky with iron core, likely has atmosphere"
    },
    {
      image: planet3,
      name: "PROXIMA B",
      type: "Rocky Planet",
      distance: "4.24 light-years",
      discoveryYear: "2016",
      mass: "1.27 Earth masses",
      radius: "1.1 Earth radii",
      orbitalPeriod: "11.2 days",
      temperature: "234 K (-39°C)",
      description: "The closest exoplanet to our solar system, Proxima B orbits within the habitable zone of Proxima Centauri, our nearest stellar neighbor. Despite being tidally locked to its star, showing the same face perpetually, it offers fascinating possibilities for life. Its proximity makes it a prime target for future interstellar missions.",
      habitability: "0.87 (Very High)",
      composition: "Rocky, possible subsurface oceans"
    },
    {
      image: planet4,
      name: "HD 40307g",
      type: "Super-Earth",
      distance: "42 light-years",
      discoveryYear: "2012",
      mass: "7.1 Earth masses",
      radius: "2.4 Earth radii",
      orbitalPeriod: "197.8 days",
      temperature: "227 K (-46°C)",
      description: "A super-Earth located 42 light-years away in the constellation Pictor, HD 40307g is believed to have conditions suitable for liquid water and potentially life. It orbits at a distance from its host star similar to Earth's distance from the Sun, receiving comparable amounts of stellar radiation. Its larger size suggests a thicker atmosphere.",
      habitability: "0.744 (High)",
      composition: "Rocky core with thick atmosphere, possible oceans"
    }
  ];

  return (
    <div className="flex-1 bg-black relative overflow-hidden">
      {/* Starfield background */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-5" />

      {/* Carousel Section */}
      <section className="relative z-10 container mx-auto px-8 py-12">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-5xl font-bold text-white mb-12 text-center">
            Potentially Habitable Exoplanets
          </h1>

          <Carousel className="w-full">
            <CarouselContent>
              {planetsData.map((planet, index) => (
                <CarouselItem key={index}>
                  <div className="grid md:grid-cols-12 gap-8 items-center">
                    {/* Planet Image */}
                    <div className="col-span-5 col-start-2 relative h-[500px] rounded-2xl overflow-hidden">
                      <img
                        src={planet.image}
                        alt={planet.name}
                        className="w-full h-full object-cover"
                        style={{
                          filter: "brightness(0.9) drop-shadow(0 0 60px rgba(59, 130, 246, 0.3))",
                        }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                    </div>

                    {/* Planet Info */}
                    <div className="col-span-5 space-y-6 text-white">
                      <div>
                        <p className="text-white/60 text-sm tracking-widest uppercase mb-2">
                          {planet.type}
                        </p>
                        <h2 className="text-5xl font-bold mb-4">{planet.name}</h2>
                      </div>

                      <p className="text-white/80 leading-relaxed">
                        {planet.description}
                      </p>

                      <div className="grid grid-cols-2 gap-4 pt-4">
                        <div className="space-y-1">
                          <p className="text-white/50 text-xs uppercase tracking-wider">Distance</p>
                          <p className="text-lg font-semibold">{planet.distance}</p>
                        </div>
                        <div className="space-y-1">
                          <p className="text-white/50 text-xs uppercase tracking-wider">Discovery</p>
                          <p className="text-lg font-semibold">{planet.discoveryYear}</p>
                        </div>
                        <div className="space-y-1">
                          <p className="text-white/50 text-xs uppercase tracking-wider">Mass</p>
                          <p className="text-lg font-semibold">{planet.mass}</p>
                        </div>
                        <div className="space-y-1">
                          <p className="text-white/50 text-xs uppercase tracking-wider">Radius</p>
                          <p className="text-lg font-semibold">{planet.radius}</p>
                        </div>
                        <div className="space-y-1">
                          <p className="text-white/50 text-xs uppercase tracking-wider">Orbital Period</p>
                          <p className="text-lg font-semibold">{planet.orbitalPeriod}</p>
                        </div>
                        <div className="space-y-1">
                          <p className="text-white/50 text-xs uppercase tracking-wider">Temperature</p>
                          <p className="text-lg font-semibold">{planet.temperature}</p>
                        </div>
                      </div>

                      <div className="pt-4 space-y-2">
                        <div className="flex justify-between items-center">
                          <p className="text-white/50 text-xs uppercase tracking-wider">Habitability Index</p>
                          <p className="text-lg font-bold text-blue-400">{planet.habitability}</p>
                        </div>
                        <div>
                          <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Composition</p>
                          <p className="text-sm text-white/70">{planet.composition}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CarouselItem>
              ))}
            </CarouselContent>
            <CarouselPrevious className="left-1 bg-white/10 border-white/20 hover:bg-white/20 text-white" />
            <CarouselNext className="right-6 bg-white/10 border-white/20 hover:bg-white/20 text-white" />
          </Carousel>
        </div>
      </section>
    </div>
  );
};

export default Planets;
