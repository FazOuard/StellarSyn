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
const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <div className="flex flex-col h-screen">
        <NavBar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/classifier" element={<Classifier />} />
          <Route path="/satellites" element={<Satellites />} />
          <Route path="/planets" element={<Planets />} />
          <Route path="/3d" element={<ExoplanetPredictor3D />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
        </div>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
