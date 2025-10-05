import { Sparkles, ArrowRight, Rocket, Upload, BarChart3 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const ExoplanetHub = () => {
  const navigate = useNavigate();

  const modes = [
    {
      id: 'single',
      title: 'Single Prediction',
      description: 'Configure parameters manually and predict one exoplanet candidate',
      icon: Sparkles,
      color: 'from-blue-500 to-purple-500',
      features: ['Interactive 3D visualization', 'Real-time predictions', 'Habitability score', 'Parameter sliders'],
      route: '/3d'
    },
    {
      id: 'batch',
      title: 'Batch Classification',
      description: 'Upload a CSV/TSV file to analyze multiple candidates at once',
      icon: Upload,
      color: 'from-green-500 to-emerald-500',
      features: ['Auto-detect mission type', 'Statistical analysis', 'ROC & PR curves', 'SHAP explanations'],
      route: '/classifier'
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden ">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px]" />
      <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-green-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "2s" }} />

      <div className="container mx-auto px-4 py-16 relative z-10">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <div className="flex justify-center mb-6">
            <div className="relative">
              <Rocket className="h-24 w-24 text-blue-400 animate-bounce" />
              <Sparkles className="h-8 w-8 text-purple-400 absolute -top-2 -right-2 animate-pulse" />
            </div>
          </div>
          
          <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Exoplanet Discovery Hub
          </h1>
          
          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-4">
            Discover and classify exoplanets using advanced machine learning models
          </p>
          
          <div className="flex justify-center gap-4 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              <BarChart3 className="h-4 w-4" />
              Kepler • K2 • TESS
            </span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <Sparkles className="h-4 w-4" />
              AI-Powered
            </span>
          </div>
        </div>

        {/* Mode Selection Cards */}
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          {modes.map((mode) => {
            const Icon = mode.icon;
            return (
              <div
                key={mode.id}
                className="group relative bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 hover:border-blue-500/50 transition-all duration-300 cursor-pointer transform hover:scale-105 hover:shadow-2xl hover:shadow-blue-500/20"
                onClick={() => navigate(mode.route)}
              >
                {/* Gradient overlay on hover */}
                <div className={`absolute inset-0 bg-gradient-to-br ${mode.color} opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity duration-300`} />
                
                <div className="relative z-10">
                  {/* Icon */}
                  <div className={`w-16 h-16 rounded-xl bg-gradient-to-br ${mode.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className="h-8 w-8 text-white" />
                  </div>

                  {/* Title */}
                  <h2 className="text-2xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors">
                    {mode.title}
                  </h2>

                  {/* Description */}
                  <p className="text-gray-400 mb-6 leading-relaxed">
                    {mode.description}
                  </p>

                  {/* Features */}
                  <ul className="space-y-2 mb-6">
                    {mode.features.map((feature, idx) => (
                      <li key={idx} className="flex items-center gap-2 text-sm text-gray-500">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                        {feature}
                      </li>
                    ))}
                  </ul>

                  {/* Button */}
                  <button className="w-full bg-slate-800 group-hover:bg-gradient-to-r group-hover:from-blue-600 group-hover:to-purple-600 text-white font-semibold py-3 px-6 rounded-lg flex items-center justify-center gap-2 transition-all duration-300">
                    Get Started
                    <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Info Section */}
        <div className="max-w-4xl mx-auto bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-500/30 rounded-xl p-8 backdrop-blur-sm">
          <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-blue-400" />
            About This Tool
          </h3>
          <p className="text-gray-300 leading-relaxed mb-4">
            Our exoplanet classification system uses a <strong className="text-blue-400">Stacking Ensemble</strong> combining Random Forest, Gradient Boosting, and AdaBoost classifiers. 
            Trained on data from NASA's Kepler, K2, and TESS missions, it achieves 96.6% average confidence with 92% of predictions at high confidence levels (≥90%).
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700/50">
              <div className="text-blue-400 font-bold text-2xl mb-1">3</div>
              <div className="text-gray-400 text-sm">Mission Datasets</div>
              <div className="text-xs text-gray-500 mt-1">Kepler • K2 • TESS</div>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700/50">
              <div className="text-purple-400 font-bold text-2xl mb-1">96.6%</div>
              <div className="text-gray-400 text-sm">Average Confidence</div>
              <div className="text-xs text-gray-500 mt-1">Stacking Ensemble</div>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700/50">
              <div className="text-green-400 font-bold text-2xl mb-1">92%</div>
              <div className="text-gray-400 text-sm">High Confidence Rate</div>
              <div className="text-xs text-gray-500 mt-1">≥90% predictions</div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.8s ease-out;
        }
      `}</style>
    </div>
  );
};

export default ExoplanetHub;