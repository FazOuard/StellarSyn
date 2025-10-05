import { Rocket, Github, Linkedin, Mail, ExternalLink, Heart, Youtube } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-slate-950 border-t border-slate-800 mt-20">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* About Section */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              
              <h3 className="text-xl font-bold text-white">STELLARSYN</h3>
            </div>
            <p className="text-gray-400 text-sm leading-relaxed">
              An AI-powered platform for exoplanet detection using machine learning models trained on NASA's Kepler, K2, and TESS mission data.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-white font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <a href="/" className="text-gray-400 hover:text-blue-400 transition-colors text-sm flex items-center gap-1">
                  Home
                </a>
              </li>
              <li>
                <a href="/3d" className="text-gray-400 hover:text-blue-400 transition-colors text-sm flex items-center gap-1">
                  Single Prediction
                </a>
              </li>
              <li>
                <a href="/classifier" className="text-gray-400 hover:text-blue-400 transition-colors text-sm flex items-center gap-1">
                  Batch Classification
                </a>
              </li>
              <li>
                <a href="https://exoplanetarchive.ipac.caltech.edu/" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-blue-400 transition-colors text-sm flex items-center gap-1">
                  NASA Exoplanet Archive <ExternalLink className="h-3 w-3" />
                </a>
              </li>
            </ul>
          </div>

          {/* Contact & Connect */}
          <div>
            <h4 className="text-white font-semibold mb-4">Connect With Us</h4>
            <div className="space-y-3">
              <a 
                href="https://github.com/FazOuard/StellarSyn" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors text-sm"
              >
                <Github className="h-5 w-5" />
                GitHub Repository
              </a>
              
              <a 
                href="https://github.com/FazOuard/StellarSyn"
                className="flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors text-sm"
              >
                <Youtube className="h-5 w-5" />
                Watch our video
              </a>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-slate-800 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-gray-500 text-sm text-center md:text-left">
              © {new Date().getFullYear()} Exoplanet Discovery Hub. Built with passion for space exploration.
            </p>
            
            <div className="flex items-center gap-2 text-gray-500 text-sm">
              <span>Made with</span>
              <Heart className="h-4 w-4 text-red-500 fill-red-500" />
              <span>by the StellarSyn's team</span>
            </div>
          </div>

          {/* Technologies Used */}
          <div className="mt-6 text-center">
            <p className="text-gray-600 text-xs mb-2">Powered by</p>
            <div className="flex flex-wrap justify-center gap-3 text-xs text-gray-500">
              <span className="px-3 py-1 bg-slate-900 rounded-full border border-slate-800">React</span>
              <span className="px-3 py-1 bg-slate-900 rounded-full border border-slate-800">FastAPI</span>
              <span className="px-3 py-1 bg-slate-900 rounded-full border border-slate-800">Machine Learning</span>
              <span className="px-3 py-1 bg-slate-900 rounded-full border border-slate-800">Three.js</span>
              <span className="px-3 py-1 bg-slate-900 rounded-full border border-slate-800">TailwindCSS</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;