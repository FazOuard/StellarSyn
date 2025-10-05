import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NavBar  from "./components/ui/navbar";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Classifier from "./pages/Classifier";
import Planets from "./pages/Planets";
import NotFound from "./pages/NotFound";
import ExoplanetPredictor3D from "./pages/ExoplanetPredictor3D";
import Satellites from "./pages/Satellites";
import Home from "./pages/Home";
import ExoplanetHub from "./pages/ExoplanetHub";
import  Footer  from "@/components/ui/footer.tsx";
import './index.css'
const queryClient = new QueryClient();

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
        
          <Routes>
            {/* Route Home en plein écran sans navbar/footer */}
            
            
            {/* Autres routes avec layout */}
            <Route path="/*" element={
              <div className="flex flex-col min-h-screen">
                <NavBar />
                <main className="flex-1 overflow-hidden">
                  
                  <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/classifier" element={<Classifier />} />
                    <Route path="/satellites" element={<Satellites />} />
                    <Route path="/planets" element={<Planets />} />
                    <Route path="/3d" element={<ExoplanetPredictor3D />} />
                    <Route path="/choose" element={<ExoplanetHub />} />
                    <Route path="*" element={<NotFound />} />
                  </Routes>
                </main>
                
              </div>
            } />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
