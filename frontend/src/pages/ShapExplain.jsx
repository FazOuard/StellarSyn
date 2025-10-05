import { useState, useEffect } from "react";
import { Loader2, Brain, TrendingUp, Info } from "lucide-react";

const ShapExplain = ({ destination, accuracy }) => {
  const [shapImageUrl, setShapImageUrl] = useState(null);
  const [isLoadingShap, setIsLoadingShap] = useState(false);

  useEffect(() => {
    if (!destination) return;
    
    const loadShapCurve = async () => {
      setIsLoadingShap(true);
      setShapImageUrl(null);
      
      try {
        const dest = destination === "auto" ? "kepler" : destination;
        const response = await fetch(`http://127.0.0.1:8000/shap-plot/${dest}`);
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const blob = await response.blob();
        setShapImageUrl(URL.createObjectURL(blob));
      } catch (err) {
        console.error("Error loading SHAP plot:", err);
        setShapImageUrl(null);
      } finally {
        setIsLoadingShap(false);
      }
    };
    
    loadShapCurve();
  }, [destination]);

  return (
    <div className="mt-8 space-y-6">
      {/*  Header Section */}
      <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-lg p-6 border border-blue-500/30">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-blue-500/20 rounded-lg">
            <Brain className="h-8 w-8 text-blue-400" />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-white mb-2">
              Model Explainability
            </h2>
            <p className="text-gray-300 text-sm leading-relaxed">
              Understanding <span className="text-blue-400 font-semibold">why</span> the model made its predictions. 
              SHAP (SHapley Additive exPlanations) values show which features had the most impact on the classification decisions.
            </p>
          </div>
        </div>
      </div>

      {/* 📊 SHAP Plot Section */}
      <div className="bg-slate-900/50 rounded-lg border border-slate-700/50 overflow-hidden">
        <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-5 w-5 text-blue-400" />
              <h3 className="text-lg font-semibold text-white">
                Feature Importance Analysis
              </h3>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Info className="h-4 w-4" />
              <span>Dataset: {destination?.toUpperCase()}</span>
            </div>
          </div>
        </div>

        <div className="p-6">
          {isLoadingShap ? (
            <div className="flex flex-col items-center justify-center py-16 space-y-4">
              <Loader2 className="h-12 w-12 animate-spin text-blue-400" />
              <p className="text-gray-400 text-sm">
                Computing SHAP values... This may take a moment
              </p>
            </div>
          ) : shapImageUrl ? (
            <div className="space-y-4">
              <img
                src={shapImageUrl}
                alt="SHAP Feature Importance"
                className="w-full max-w-2xl mx-auto rounded-lg shadow-2xl"
              />
              
              {/* 💡 Insights Section */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <InsightCard
                  icon="🎯"
                  title="Most Important"
                  description="Top features drive model predictions"
                  color="blue"
                />
                <InsightCard
                  icon="⚖️"
                  title="Balanced Analysis"
                  description="SHAP considers all feature interactions"
                  color="purple"
                />
                <InsightCard
                  icon="📈"
                  title="Model Accuracy"
                  description={`${(accuracy * 100).toFixed(1)}% on test data`}
                  color="green"
                />
              </div>

              {/* 📝 Interpretation Guide */}
              <div className="mt-6 p-4 bg-slate-800/30 rounded-lg border border-slate-700/30">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <Info className="h-4 w-4 text-blue-400" />
                  How to Read This Chart
                </h4>
                <ul className="text-gray-300 text-sm space-y-1 ml-6 list-disc">
                  <li>Longer bars = higher impact on predictions</li>
                  <li>Features at the top influenced the model most</li>
                  <li>SHAP values measure average contribution to output</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="text-center py-16">
              <Brain className="h-16 w-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">
                SHAP analysis will appear after classification
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// 💎 Insight Card Component
const InsightCard = ({ icon, title, description, color }) => {
  const colorClasses = {
    blue: "bg-blue-500/10 border-blue-500/30 text-blue-400",
    purple: "bg-purple-500/10 border-purple-500/30 text-purple-400",
    green: "bg-green-500/10 border-green-500/30 text-green-400"
  };

  return (
    <div className={`p-4 rounded-lg border ${colorClasses[color]}`}>
      <div className="text-2xl mb-2">{icon}</div>
      <h5 className="font-semibold mb-1">{title}</h5>
      <p className="text-xs text-gray-400">{description}</p>
    </div>
  );
};

export default ShapExplain;