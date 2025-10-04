import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";

const ShapExplain = ({ destination }) => {
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
    <div className="p-6 bg-slate-900/50 rounded-lg mt-6 text-center">
      <h3 className="text-lg font-semibold text-white mb-4">SHAP Summary Plot</h3>
      {isLoadingShap ? (
        <div className="flex justify-center items-center text-white">
          <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading SHAP...
        </div>
      ) : shapImageUrl ? (
        <img
          src={shapImageUrl}
          alt="SHAP Summary Plot"
          style={{ maxWidth: "100%", height: "auto", borderRadius: "8px" }}
        />
      ) : (
        <p className="text-gray-400">The SHAP plot will appear after classification.</p>
      )}
    </div>
  );
};

export default ShapExplain;
