import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from "@/components/ui/carousel";
import Kepler from "@/assets/kepler3.jpg";
import Kepler2 from "@/assets/k2.jpg";
import Tess from "@/assets/tess.jpg";

const Satellites = () => {
  const satellitesData = [
    {
      image: Kepler,
      name: "Kepler Space Telescope",
      type: "NASA Exoplanet Mission",
      launchDate: "March 7, 2009",
      missionEnd: "October 30, 2018",
      mass: "1,039 kg",
      orbit: "Heliocentric (Earth-trailing)",
      agency: "NASA Ames Research Center",
      description:
        "The Kepler Space Telescope revolutionized our understanding of the universe by discovering thousands of exoplanets. Its mission focused on finding Earth-sized planets orbiting within the habitable zones of Sun-like stars using the transit method.",
      achievements: "Discovered over 2,600 confirmed exoplanets",
      power: "Solar-powered (1100W)",
    },
    {
      image: Kepler2,
      name: "K2 Mission",
      type: "Extended Exoplanet Mission",
      launchDate: "May 30, 2014",
      missionEnd: "October 2018",
      mass: "1,039 kg (same spacecraft as Kepler)",
      orbit: "Heliocentric (Earth-trailing)",
      agency: "NASA Ames Research Center",
      description:
        "After Kepler's original mission ended due to reaction wheel failure, NASA repurposed the telescope for the K2 mission. K2 observed different regions of the sky and contributed to exoplanet discoveries, stellar studies, and supernova research.",
      achievements: "Discovered 500+ exoplanets and thousands of candidates",
      power: "Solar-powered (800–1000W)",
    },
    {
      image: Tess,
      name: "TESS (Transiting Exoplanet Survey Satellite)",
      type: "NASA Exoplanet Survey Satellite",
      launchDate: "April 18, 2018",
      missionEnd: "Active",
      mass: "362 kg",
      orbit: "Highly elliptical Earth orbit (P/2 Lunar Resonance)",
      agency: "NASA & MIT",
      description:
        "TESS continues Kepler's legacy by surveying nearly the entire sky to identify planets around bright, nearby stars. Its discoveries are ideal for follow-up studies by telescopes like the James Webb Space Telescope (JWST).",
      achievements: "Over 400 confirmed exoplanets and 6,000 candidates",
      power: "Solar-powered (400W)",
    },
  ];

  return (
    <div className="flex-1 bg-black relative overflow-hidden">
      {/* Background grid */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-5" />

      {/* Carousel Section */}
      <section className="relative z-10 container mx-auto px-8 py-12">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-5xl font-bold text-white mb-12 text-center">
            Notable Exoplanet-Hunting Satellites
          </h1>

          <Carousel className="w-full">
            <CarouselContent>
              {satellitesData.map((satellite, index) => (
                <CarouselItem key={index}>
                  <div className="grid md:grid-cols-12 gap-8 items-center">
                    {/* Satellite Image */}
                    <div className="col-span-5 col-start-2 relative h-[500px] rounded-2xl overflow-hidden">
                      <img
                        src={satellite.image}
                        alt={satellite.name}
                        className="w-full h-full object-cover"
                        style={{
                          filter: "brightness(0.9) drop-shadow(0 0 60px rgba(59,130,246,0.3))",
                        }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                    </div>

                    {/* Satellite Info */}
                    <div className="col-span-5 space-y-6 text-white">
                      <div>
                        <p className="text-white/60 text-sm tracking-widest uppercase mb-2">
                          {satellite.type}
                        </p>
                        <h2 className="text-5xl font-bold mb-4">{satellite.name}</h2>
                      </div>

                      <p className="text-white/80 leading-relaxed">
                        {satellite.description}
                      </p>

                      <div className="grid grid-cols-2 gap-4 pt-4">
                        <div>
                          <p className="text-white/50 text-xs uppercase tracking-wider">Launch Date</p>
                          <p className="text-lg font-semibold">{satellite.launchDate}</p>
                        </div>
                        <div>
                          <p className="text-white/50 text-xs uppercase tracking-wider">Mission End</p>
                          <p className="text-lg font-semibold">{satellite.missionEnd}</p>
                        </div>
                        <div>
                          <p className="text-white/50 text-xs uppercase tracking-wider">Mass</p>
                          <p className="text-lg font-semibold">{satellite.mass}</p>
                        </div>
                        <div>
                          <p className="text-white/50 text-xs uppercase tracking-wider">Orbit</p>
                          <p className="text-lg font-semibold">{satellite.orbit}</p>
                        </div>
                        <div>
                          <p className="text-white/50 text-xs uppercase tracking-wider">Agency</p>
                          <p className="text-lg font-semibold">{satellite.agency}</p>
                        </div>
                        <div>
                          <p className="text-white/50 text-xs uppercase tracking-wider">Power</p>
                          <p className="text-lg font-semibold">{satellite.power}</p>
                        </div>
                      </div>

                      <div className="pt-4 space-y-2">
                        <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Major Achievement</p>
                        <p className="text-sm text-white/70">{satellite.achievements}</p>
                      </div>
                    </div>
                  </div>
                </CarouselItem>
              ))}
            </CarouselContent>

            {/* Carousel Controls */}
            <CarouselPrevious className="left-1 bg-white/10 border-white/20 hover:bg-white/20 text-white" />
            <CarouselNext className="right-6 bg-white/10 border-white/20 hover:bg-white/20 text-white" />
          </Carousel>
        </div>
      </section>
    </div>
  );
};

export default Satellites;
