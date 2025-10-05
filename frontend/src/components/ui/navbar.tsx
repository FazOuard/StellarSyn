import { Link } from "react-router-dom";

const NavBar = () => {
  return (
    <nav className="h-[10%] bg-black px-8 py-8">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <Link to="/" className="text-3xl md:text-4xl font-bold tracking-wider text-white">
            StellarSyn
          </Link>
        </div>
        <div className="flex gap-8 text-sm text-white/80">
          <Link to="/" className="hover:text-white transition-colors text-lg">
           Home
          </Link>
          <Link to="/satellites" className="hover:text-white transition-colors text-lg">
            Satellites
          </Link>
          <Link to="/planets" className="hover:text-white transition-colors text-lg">
            Planets
          </Link>
          <Link to="/about" className="hover:text-white transition-colors text-lg">
            About Us
          </Link>
          <Link to="/choose" className="hover:text-white transition-colors text-lg">
            Predict
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
