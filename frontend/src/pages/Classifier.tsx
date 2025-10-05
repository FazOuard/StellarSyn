import { useState, useEffect } from "react";
import { ArrowLeft, Loader2, Sparkles, Upload, CheckCircle,Info, XCircle,Brain, AlertCircle, TrendingUp, BarChart3 } from "lucide-react";
import ShapExplain from "./ShapExplain"; 

const Classifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [stats, setStats] = useState(null);
  const [notification, setNotification] = useState(null);
  const [destination, setDestination] = useState("auto");
  const [rocImageUrl, setRocImageUrl] = useState(null);
  const [isLoadingRoc, setIsLoadingRoc] = useState(false);
  const [prImageUrl, setPrImageUrl] = useState(null);
  const [isLoadingPr, setIsLoadingPr] = useState(false);

  // Classification labels mapping (votre mapping d’origine)
  const classificationSchemes = {
    kepler: {
      0: { name: "False Positive", icon: XCircle, color: "text-red-500", bg: "bg-red-500/10", barColor: "bg-red-500" },
      1: { name: "Confirmed Planet", icon: CheckCircle, color: "text-green-500", bg: "bg-green-500/10", barColor: "bg-green-500" },
      2: { name: "Candidate", icon: AlertCircle, color: "text-yellow-500", bg: "bg-yellow-500/10", barColor: "bg-yellow-500" }
    },
    k2: {
      0: { name: "Confirmed", icon: CheckCircle, color: "text-green-500", bg: "bg-green-500/10", barColor: "bg-green-500" },
      1: { name: "Candidate", icon: AlertCircle, color: "text-yellow-500", bg: "bg-yellow-500/10", barColor: "bg-yellow-500" },
      2: { name: "False Positive", icon: XCircle, color: "text-red-500", bg: "bg-red-500/10", barColor: "bg-red-500" },
      3: { name: "Refuted", icon: XCircle, color: "text-orange-500", bg: "bg-orange-500/10", barColor: "bg-orange-500" }
    },
    toi: {
      0: { name: "False Positive (FP)", icon: XCircle, color: "text-red-500", bg: "bg-red-500/10", barColor: "bg-red-500" },
      1: { name: "Planetary Candidate (PC)", icon: AlertCircle, color: "text-yellow-500", bg: "bg-yellow-500/10", barColor: "bg-yellow-500" },
      2: { name: "Known Planet (KP)", icon: CheckCircle, color: "text-green-500", bg: "bg-green-500/10", barColor: "bg-green-500" },
      3: { name: "Ambiguous (APC)", icon: AlertCircle, color: "text-orange-500", bg: "bg-orange-500/10", barColor: "bg-orange-500" },
      4: { name: "False Alarm (FA)", icon: XCircle, color: "text-pink-500", bg: "bg-pink-500/10", barColor: "bg-pink-500" },
      5: { name: "Confirmed Planet (CP)", icon: CheckCircle, color: "text-emerald-500", bg: "bg-emerald-500/10", barColor: "bg-emerald-500" }
    }
  };

  const getClassLabel = (classId, destination) => {
    const scheme = classificationSchemes[destination] || classificationSchemes.kepler;
    return scheme[classId] || {
      name: `Class ${classId}`,
      icon: AlertCircle,
      color: "text-gray-500",
      bg: "bg-gray-500/10",
      barColor: "bg-gray-500"
    };
  };

  const showToast = (title, description, variant = "default") => {
    setNotification({ title, description, variant });
    setTimeout(() => setNotification(null), 5000);
  };

  // Gestion upload fichier
  const handleFileUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
      showToast("File too large", "Please select a file under 5MB", "destructive");
      return;
    }

    setSelectedFile(file);
    setStats(null);
    setRocImageUrl(null); // reset courbe ROC quand nouveau fichier
  };

  // Calcul stats de classification
  const calculateStats = (predictions) => {
    try {
      const total = predictions.length;
      const classCounts = {};
      let totalConfidence = 0;
      let highConfidence = 0;

      predictions.forEach(pred => {
        const predValue = pred.prediction;
        classCounts[predValue] = (classCounts[predValue] || 0) + 1;
        totalConfidence += (pred.probability || 0);
        if ((pred.probability || 0) >= 0.9) highConfidence++;
      });

      const avgConfidence = total > 0 ? totalConfidence / total : 0;

      return {
        total,
        classCounts,
        avgConfidence,
        highConfidence,
        highConfidencePercent: total > 0 ? (highConfidence / total) * 100 : 0
      };
    } catch (error) {
      console.error("Error calculating stats:", error);
      return {
        total: 0,
        classCounts: {},
        avgConfidence: 0,
        highConfidence: 0,
        highConfidencePercent: 0
      };
    }
  };

  // Récupération automatique de la courbe ROC après classification
  useEffect(() => {
    if (!stats?.detectedDestination) return;

    const loadCurves = async () => {
      const dest = stats.detectedDestination === "auto" ? "kepler" : stats.detectedDestination;

      // ROC Curve
      setIsLoadingRoc(true);
      setRocImageUrl(null);
      try {
        const response = await fetch(`http://127.0.0.1:8000/roc-curve/${dest}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const blob = await response.blob();
        setRocImageUrl(URL.createObjectURL(blob));
      } catch (err) {
        console.error("Error loading ROC curve:", err);
        setRocImageUrl(null);
      } finally {
        setIsLoadingRoc(false);
      }

      // PR Curve
      setIsLoadingPr(true);
      setPrImageUrl(null);
      try {
        const response = await fetch(`http://127.0.0.1:8000/pr-curve/${dest}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const blob = await response.blob();
        setPrImageUrl(URL.createObjectURL(blob));
      } catch (err) {
        console.error("Error loading PR curve:", err);
        setPrImageUrl(null);
      } finally {
        setIsLoadingPr(false);
      }
    };

    loadCurves();
  }, [stats]);

  const [f1ImageUrl, setF1ImageUrl] = useState(null);
const [isLoadingF1, setIsLoadingF1] = useState(false);

// Fonction pour charger la F1 curve
useEffect(() => {
  if (!stats?.detectedDestination) return;

  const loadF1Curve = async () => {
    setIsLoadingF1(true);
    setF1ImageUrl(null);
    try {
      const dest = stats.detectedDestination === "auto" ? "kepler" : stats.detectedDestination;
      const response = await fetch(`http://127.0.0.1:8000/confidence-distribution/${dest}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const blob = await response.blob();
      setF1ImageUrl(URL.createObjectURL(blob));
    } catch (err) {
      console.error("Error loading F1 curve:", err);
      setF1ImageUrl(null);
    } finally {
      setIsLoadingF1(false);
    }
  };

  loadF1Curve();
}, [stats]);


  // Classification
  const handleClassify = async () => {
    if (!selectedFile) return;

    setIsClassifying(true);
    setStats(null);
    setRocImageUrl(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      let url = `http://127.0.0.1:8000/predict-file?has_labels=false`;
      if (destination !== "auto") {
        url += `&destination=${destination}`;
      }

      const response = await fetch(url, { method: "POST", body: formData });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.predictions && Array.isArray(data.predictions)) {
        const calculatedStats = calculateStats(data.predictions);
        calculatedStats.detectedDestination = data.destination;
        setStats(calculatedStats);

        const destName = data.destination === 'toi' ? 'TESS' : data.destination.toUpperCase();
        showToast("Classification complete!", `Successfully analyzed ${calculatedStats.total} samples using ${destName} model`);
      } else {
        throw new Error("Invalid response format: predictions array not found");
      }
    } catch (error) {
      console.error("Prediction error:", error);
      showToast("Classification failed", error.message || "Please try again later", "destructive");
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden overflow-y-auto">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px]" />
      <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />

      {/* Toast Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-md animate-fade-in ${notification.variant === "destructive" ? "bg-red-900/90 border border-red-700" : "bg-slate-800/90 border border-slate-700"}`}>
          <h4 className="font-semibold text-white mb-1">{notification.title}</h4>
          <p className="text-sm text-gray-300">{notification.description}</p>
        </div>
      )}

      <div className="container mx-auto px-4 py-8 relative z-10">
        <button onClick={() => window.history.back()} className="mb-8 text-gray-300 hover:text-blue-400 transition-colors flex items-center gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back to Home
        </button>

        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Exoplanet Detector
            </h1>
            <p className="text-xl text-gray-400">
              Upload a CSV/TSV dataset to predict exoplanet candidates
            </p>
          </div>

          <div className="p-8 bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-lg shadow-lg mb-6">
            <div className="space-y-6">
              {/* Destination Selector */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Dataset Type
                </label>
                <select
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  disabled={isClassifying}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                >
                  <option value="auto">Auto-detect (Recommended)</option>
                  <option value="kepler">Kepler Mission</option>
                  <option value="k2">K2 Mission</option>
                  <option value="toi">TESS (TOI)</option>
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  Auto-detect will analyze your file and choose the best model automatically
                </p>
              </div>

              {/* Upload Area */}
              <div className="relative">
                <input
                  type="file"
                  accept=".csv,.tsv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  disabled={isClassifying}
                />
                <label
                  htmlFor="file-upload"
                  className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-slate-700 rounded-lg cursor-pointer hover:border-blue-500 transition-colors bg-slate-800/20 ${isClassifying ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  {selectedFile ? (
                    <div className="flex flex-col items-center">
                      <CheckCircle className="w-12 h-12 mb-4 text-green-500" />
                      <p className="text-white font-semibold">{selectedFile.name}</p>
                      <p className="text-xs text-gray-400 mt-2">
                        {(selectedFile.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-12 h-12 mb-4 text-blue-400" />
                      <p className="mb-2 text-sm text-white">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-gray-400">CSV or TSV (max 5MB)</p>
                    </div>
                  )}
                </label>
              </div>

              {/* Classify Button */}
              <button
                onClick={handleClassify}
                disabled={!selectedFile || isClassifying}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all flex items-center justify-center gap-2"
              >
                {isClassifying ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5" />
                    Detect Exoplanet
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Statistics Cards */}
          {stats && (
            <div className="space-y-6 animate-fade-in">
              {/* Overview Card */}
              <div className="p-6 bg-gradient-to-br from-blue-900/30 to-purple-900/30 border border-blue-700/30 backdrop-blur-sm rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <BarChart3 className="h-6 w-6 text-blue-400 mr-2" />
                    <h2 className="text-2xl font-bold text-white">Classification Results</h2>
                  </div>
                  <div className="px-3 py-1 bg-blue-500/20 border border-blue-500/30 rounded-full text-sm text-blue-300">
                    {stats.detectedDestination ? stats.detectedDestination.toUpperCase() : destination.toUpperCase()}
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-slate-900/50 p-4 rounded-lg">
                    <p className="text-sm text-gray-400 mb-1">Total Samples</p>
                    <p className="text-3xl font-bold text-white">{stats.total}</p>
                  </div>
                  
                  <div className="bg-slate-900/50 p-4 rounded-lg">
                    <p className="text-sm text-gray-400 mb-1">Avg Confidence</p>
                    <p className="text-3xl font-bold text-blue-400">
                      {(stats.avgConfidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  
                  <div className="bg-slate-900/50 p-4 rounded-lg">
                    <p className="text-sm text-gray-400 mb-1">High Confidence</p>
                    <p className="text-3xl font-bold text-green-500">
                      {stats.highConfidencePercent.toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      ({stats.highConfidence} samples ≥ 90%)
                    </p>
                  </div>
                </div>

                {/* Distribution by Class */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-white flex items-center">
                    <TrendingUp className="h-5 w-5 mr-2 text-blue-400" />
                    Distribution by Classification
                  </h3>
                  
                  {Object.entries(stats.classCounts).map(([classId, count]) => {
                    const label = getClassLabel(classId, stats.detectedDestination || "kepler");
                    const percentage = stats.total > 0 ? (count / stats.total) * 100 : 0;
                    const Icon = label.icon;
                    
                    return (
                      <div key={classId} className={`${label.bg} p-4 rounded-lg border border-slate-700/50`}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center">
                            <Icon className={`h-5 w-5 ${label.color} mr-2`} />
                            <span className="font-semibold text-white">{label.name}</span>
                          </div>
                          <div className="text-right">
                            <span className="text-lg font-bold text-white">{count}</span>
                            <span className={`ml-2 text-sm ${label.color}`}>
                              ({percentage.toFixed(1)}%)
                            </span>
                          </div>
                        </div>
                        
                        <div className="w-full bg-slate-800/50 rounded-full h-3 overflow-hidden">
                          <div
                            className={`h-3 rounded-full ${label.barColor} transition-all duration-1000 ease-out`}
                            style={{ 
                              width: `${percentage}%`,
                              animation: 'slideIn 1s ease-out'
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-lg p-6 border border-blue-500/30">
                      <div className="flex items-start gap-4">
                        <div className="p-3 bg-blue-500/20 rounded-lg">
                          <Brain className="h-8 w-8 text-blue-400" />
                        </div>
                        <div className="flex-1">
                          <h2 className="text-2xl font-bold text-white mb-2">
                            Statistics
                          </h2>
                          <p className="text-gray-300 text-sm leading-relaxed">
                            Analyzing <span className="text-blue-400 font-semibold">model performance</span> helps assess its reliability. 
                            The <span className="text-blue-400 font-semibold">ROC curve</span> visualizes the trade-off between sensitivity and specificity, 
                            while the <span className="text-blue-400 font-semibold">Precision-Recall curve</span> highlights the balance between precision and recall for different thresholds. 
                            The <span className="text-blue-400 font-semibold">Model Confidence</span> plot shows the probability distribution of predictions across classes.
                          </p>


                        </div>
                      </div>
                    </div>   
                
              {/* ROC & PR Curves côte à côte */}
              <div className="flex flex-col md:flex-row gap-6 mt-6">
                {/* ROC Curve */}
                <div className="flex-1 p-6 bg-slate-900/50 rounded-lg text-center">
                  <h3 className="text-lg font-semibold text-white mb-4">ROC curve</h3>
                  {isLoadingRoc ? (
                    <div className="flex justify-center items-center text-white">
                      <Loader2 className="h-6 w-6 animate-spin mr-2" /> Download...
                    </div>
                  ) : rocImageUrl ? (
                    <img
                      src={rocImageUrl}
                      alt="ROC Curve"
                      style={{ maxWidth: "100%", height: "auto", borderRadius: "8px" }}
                    />
                  ) : (
                    <p className="text-gray-400">The ROC curve will appear after classification.</p>
                  )}
                </div>

                {/* PR Curve */}
                <div className="flex-1 p-6 bg-slate-900/50 rounded-lg text-center">
                  <h3 className="text-lg font-semibold text-white mb-4">Precision-Recall curve</h3>
                  {isLoadingPr ? (
                    <div className="flex justify-center items-center text-white">
                      <Loader2 className="h-6 w-6 animate-spin mr-2" /> Download...
                    </div>
                  ) : prImageUrl ? (
                    <img
                      src={prImageUrl}
                      alt="PR Curve"
                      style={{ maxWidth: "100%", height: "auto", borderRadius: "8px" }}
                    />
                  ) : (
                    <p className="text-gray-400">The PR curve will appear after classification.</p>
                  )}
                </div>
              </div>

              {/* model confidence yes*/}
              <div className="p-6 bg-slate-900/50 rounded-lg mt-6 text-center w-full md:w-3/4 lg:w-1/2 mx-auto">
              <h3 className="text-lg font-semibold text-white mb-4">Model confidence: probability of distribution</h3>
              {isLoadingF1 ? (
                <div className="flex justify-center items-center text-white">
                  <Loader2 className="h-6 w-6 animate-spin mr-2" /> Download...
                </div>
              ) : f1ImageUrl ? (
                <img
                  src={f1ImageUrl}
                  alt="Model confidence Curve"
                  style={{ maxWidth: "100%", height: "auto", borderRadius: "8px" }}
                />
              ) : (
                <p className="text-gray-400">The Model confidence curve will appear after classification.</p>
              )}
            </div>

              <div className="mt-6 p-4 bg-slate-800/30 rounded-lg border border-slate-700/30">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <Info className="h-4 w-4 text-blue-400" />
                  How to Read The Charts
                </h4>
                <ul className="text-gray-300 text-sm space-y-1 ml-6 list-disc">
                  <li><span className="font-semibold">ROC Curve:</span> The closer the curve follows the top-left corner, the better the model distinguishes between classes.</li>
                  <li><span className="font-semibold">Precision-Recall Curve:</span> Higher area under the curve indicates better precision for high recall values, useful for imbalanced datasets.</li>
                  <li><span className="font-semibold">Model Confidence:</span> Peaks near 1 indicate confident predictions; peaks near 0.5 indicate uncertainty.</li>
                </ul>
              </div>

              <ShapExplain destination={stats.detectedDestination || "kepler"} />
            

            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
        @keyframes slideIn {
          from {
            width: 0;
          }
          to {
            width: 100%;
          }
        }
      `}</style>
    </div>
  );
};

export default Classifier;
