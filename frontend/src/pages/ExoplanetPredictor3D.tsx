import { useState, useEffect } from 'react';
import { Rocket, Sparkles, Globe, Play, RotateCcw } from 'lucide-react';
import StellarSystem3D from './StellarSystem3D';
const featureKeys = {
  kepler: ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq", "koi_insol", "koi_steff", "koi_srad"],
  k2: ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "pl_eqt", "pl_insol", "st_teff", "st_rad"],
  tess: ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "pl_eqt", "pl_insol", "st_teff", "st_rad"]
};

const featureDefaults = {
  kepler: { koi_period: 10.0, koi_duration: 3.5, koi_depth: 500, koi_prad: 2.5, koi_teq: 300, koi_insol: 1.2, koi_steff: 5800, koi_srad: 1.0 },
  k2: { pl_orbper: 10.0, pl_trandurh: 3.5, pl_trandep: 500, pl_rade: 2.5, pl_eqt: 300, pl_insol: 1.2, st_teff: 5800, st_rad: 1.0 },
  tess: { pl_orbper: 10.0, pl_trandurh: 3.5, pl_trandep: 500, pl_rade: 2.5, pl_eqt: 300, pl_insol: 1.2, st_teff: 5800, st_rad: 1.0 }
};

const featureConfigLabels = {
  koi_period: { label: 'Orbital Period', unit: 'days', min: 0.5, max: 500, step: 0.5, icon: '🔄' },
  koi_duration: { label: 'Transit Duration', unit: 'hours', min: 0.5, max: 24, step: 0.1, icon: '⏱️' },
  koi_depth: { label: 'Transit Depth', unit: 'ppm', min: 10, max: 50000, step: 10, icon: '📉' },
  koi_prad: { label: 'Planetary Radius', unit: 'R⊕', min: 0.1, max: 30, step: 0.1, icon: '🪐' },
  koi_teq: { label: 'Equilibrium Temp', unit: 'K', min: 100, max: 3000, step: 10, icon: '🌡️' },
  koi_insol: { label: 'Insolation Flux', unit: 'S⊕', min: 0.01, max: 1000, step: 0.1, icon: '☀️' },
  koi_steff: { label: 'Stellar Temperature', unit: 'K', min: 3000, max: 10000, step: 100, icon: '⭐' },
  koi_srad: { label: 'Stellar Radius', unit: 'R☉', min: 0.1, max: 5, step: 0.1, icon: '🌟' },

  pl_orbper: { label: 'Orbital Period', unit: 'days', min: 0.5, max: 500, step: 0.5, icon: '🔄' },
  pl_trandurh: { label: 'Transit Duration', unit: 'hours', min: 0.5, max: 24, step: 0.1, icon: '⏱️' },
  pl_trandep: { label: 'Transit Depth', unit: 'ppm', min: 10, max: 50000, step: 10, icon: '📉' },
  pl_rade: { label: 'Planetary Radius', unit: 'R⊕', min: 0.1, max: 30, step: 0.1, icon: '🪐' },
  pl_eqt: { label: 'Equilibrium Temp', unit: 'K', min: 100, max: 3000, step: 10, icon: '🌡️' },
  pl_insol: { label: 'Insolation Flux', unit: 'S⊕', min: 0.01, max: 1000, step: 0.1, icon: '☀️' },
  st_teff: { label: 'Stellar Temperature', unit: 'K', min: 3000, max: 10000, step: 100, icon: '⭐' },
  st_rad: { label: 'Stellar Radius', unit: 'R☉', min: 0.1, max: 5, step: 0.1, icon: '🌟' }
};

// Mapping des catégories
const categories = {
  kepler: {
    0: { name: "False Positive", color: "text-red-500" },
    1: { name: "Confirmed Planet", color: "text-green-500" },
    2: { name: "Candidate", color: "text-yellow-500" }
  },
  k2: {
    0: { name: "Confirmed", color: "text-green-500" },
    1: { name: "Candidate", color: "text-yellow-500" },
    2: { name: "False Positive", color: "text-red-500" },
    3: { name: "Refuted", color: "text-orange-500" }
  },
  toi: {
    0: { name: "False Positive (FP)", color: "text-red-500" },
    1: { name: "Planetary Candidate (PC)", color: "text-yellow-500" },
    2: { name: "Known Planet (KP)", color: "text-green-500" },
    3: { name: "Ambiguous (APC)", color: "text-orange-500" },
    4: { name: "False Alarm (FA)", color: "text-pink-500" },
    5: { name: "Confirmed Planet (CP)", color: "text-emerald-500" }
  }
};

const ExoplanetPredictor3D = () => {
  const [destination, setDestination] = useState('kepler');
  const [features, setFeatures] = useState(featureDefaults.kepler);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [show3D, setShow3D] = useState(false);

  useEffect(() => {
    setFeatures(featureDefaults[destination]);
  }, [destination]);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const backendDestination = destination === 'tess' ? 'toi' : destination;

      const response = await fetch('http://127.0.0.1:8000/predict-simple', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ destination: backendDestination, simple_features: features })
      });

      if (!response.ok) throw new Error(`Error: ${response.statusText}`);

      const data = await response.json();
      setPrediction(data);
      setShow3D(true);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed. Please check your backend.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFeatures(featureDefaults[destination]);
    setPrediction(null);
    setShow3D(false);
  };

  const calculateHabitability = () => {
    let tempKey, radiusKey, fluxKey;
    if (destination === 'kepler') {
      [radiusKey, , tempKey, fluxKey] = featureKeys.kepler;
    } else {
      [radiusKey, , tempKey, fluxKey] = featureKeys[destination];
    }

    const temp = features[tempKey];
    const radius = features[radiusKey];
    const flux = features[fluxKey];

    let score = 0;
    if (temp >= 273 && temp <= 373) score += 40;
    else if (temp >= 200 && temp <= 400) score += 20;

    if (radius >= 0.5 && radius <= 2.5) score += 30;
    else if (radius >= 0.3 && radius <= 5) score += 15;

    if (flux >= 0.25 && flux <= 4) score += 30;
    else if (flux >= 0.1 && flux <= 10) score += 15;

    return Math.min(score, 100);
  };

  const currentFeatureKeys = featureKeys[destination];
  const featureConfig = currentFeatureKeys.map((key) => ({
    key,
    ...featureConfigLabels[key]
  }));

  // Determine category
  const backendDestination = destination === 'tess' ? 'toi' : destination;
  const category = prediction ? categories[backendDestination][prediction.prediction] : null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-3 mb-4">
            
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Exoplanet Discovery Lab
            </h1>
          </div>
          <p className="text-gray-400 text-lg">
            Configure planet parameters and predict with AI • Visualize in 3D
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel */}
          <div className="space-y-6">
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50">
              <label className="block text-sm font-semibold text-gray-300 mb-3">
                Select Mission Dataset
              </label>
              <select
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
              >
                <option value="kepler">Kepler Mission</option>
                <option value="k2">K2 Mission</option>
                <option value="tess">TESS Mission</option>
              </select>
            </div>

            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50 space-y-4">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-yellow-400" /> Planet Parameters
              </h3>

              {featureConfig.map((config) => (
                <div key={config.key} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                      <span>{config.icon}</span> {config.label}
                    </label>
                    <span className="text-blue-400 font-mono text-sm">
                      {features[config.key]} {config.unit}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={config.min}
                    max={config.max}
                    step={config.step}
                    value={features[config.key]}
                    onChange={(e) => setFeatures({ ...features, [config.key]: parseFloat(e.target.value) })}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
              ))}
            </div>

            <div className="flex gap-4">
              <button
                onClick={handlePredict}
                disabled={loading}
                className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-4 rounded-xl font-semibold text-lg flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-500/30"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" /> Predict & Visualize
                  </>
                )}
              </button>

              <button
                onClick={resetForm}
                className="bg-slate-700 hover:bg-slate-600 px-6 py-4 rounded-xl font-semibold flex items-center gap-2 transition-all"
              >
                <RotateCcw className="h-5 w-5" /> Reset
              </button>
            </div>
          </div>

          {/* Right Panel */}
          <div className="space-y-6">
            {prediction && (
              <div className="bg-gradient-to-br from-slate-900/80 to-blue-900/30 rounded-xl p-6 border-2 border-blue-500/50">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <Globe className="h-6 w-6 text-green-400" /> Prediction Results
                </h3>

                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                    <span className="text-gray-300">Classification:</span>
                    <span className={`text-xl font-bold ${category.color}`}>
                      {category.name}
                    </span>
                  </div>

                  {prediction.probability && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm text-gray-400">
                        <span>Confidence Level</span>
                        <span>{(prediction.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-green-500 h-full rounded-full transition-all duration-500"
                          style={{ width: `${prediction.probability * 100}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {prediction.habitability_score && (
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-gray-300">Habitability Score:</span>
                        <span className="text-xl font-bold text-yellow-400">
                          {prediction.habitability_score}/100
                        </span>
                      </div>
                      <div className="w-full bg-slate-700 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-full rounded-full transition-all duration-500"
                          style={{ width: `${prediction.habitability_score}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

           {show3D && (
            <div key={destination} className="h-96 w-full">
                <StellarSystem3D planetData={{
                stellarRadius: features[featureKeys[destination][3]],
                stellarTemp: features[featureKeys[destination][1]], 
                planetRadius: features[featureKeys[destination][3]],
                orbitalPeriod: features[featureKeys[destination][0]],
                equilibriumTemp: features[featureKeys[destination][2]]
                }} />
            </div>
            )}



            {!prediction && (
              <div className="bg-slate-900/30 rounded-xl p-6 border border-slate-700/30">
                <h4 className="font-semibold mb-3 text-blue-400">How it works:</h4>
                <ul className="space-y-2 text-sm text-gray-400">
                  <li className="flex items-start gap-2"><span className="text-blue-400 mt-1">1.</span> Adjust planetary and stellar parameters using the sliders</li>
                  <li className="flex items-start gap-2"><span className="text-blue-400 mt-1">2.</span> Click "Predict & Visualize" to run AI classification</li>
                  <li className="flex items-start gap-2"><span className="text-blue-400 mt-1">3.</span> View prediction results and habitability score</li>
                  <li className="flex items-start gap-2"><span className="text-blue-400 mt-1">4.</span> Explore the 3D visualization of the stellar system</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExoplanetPredictor3D;
