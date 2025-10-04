import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, ArrowLeft, Sparkles, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";

const Classifier = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please select an image under 5MB",
          variant: "destructive",
        });
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) return;

    setIsClassifying(true);
    setResult(null);

    try {
      const { data, error } = await supabase.functions.invoke("classify-exoplanet", {
        body: { imageData: selectedImage },
      });

      if (error) throw error;

      if (data?.result) {
        setResult(data.result);
        toast({
          title: "Classification complete",
          description: "Analysis results are ready",
        });
      }
    } catch (error) {
      console.error("Classification error:", error);
      toast({
        title: "Classification failed",
        description: "Please try again later",
        variant: "destructive",
      });
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-background/95 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
      <div className="absolute top-20 left-20 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
      <div className="absolute bottom-20 right-20 w-96 h-96 bg-secondary/20 rounded-full blur-3xl" />

      <div className="container mx-auto px-4 py-8 relative z-10">
        <Button
          variant="ghost"
          onClick={() => navigate("/")}
          className="mb-8 text-foreground hover:text-primary transition-colors"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Home
        </Button>

        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Exoplanet Classifier
            </h1>
            <p className="text-xl text-muted-foreground">
              Upload an astronomical image to detect if it contains an exoplanet
            </p>
          </div>

          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 shadow-lg">
            <div className="space-y-6">
              {/* Upload Area */}
              <div className="relative">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  id="image-upload"
                />
                <label
                  htmlFor="image-upload"
                  className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors bg-muted/20"
                >
                  {selectedImage ? (
                    <img
                      src={selectedImage}
                      alt="Selected"
                      className="max-h-full max-w-full object-contain rounded-lg"
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-12 h-12 mb-4 text-primary" />
                      <p className="mb-2 text-sm text-foreground">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-muted-foreground">PNG, JPG (max 5MB)</p>
                    </div>
                  )}
                </label>
              </div>

              {/* Classify Button */}
              <Button
                onClick={handleClassify}
                disabled={!selectedImage || isClassifying}
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground shadow-[0_0_20px_hsl(var(--primary)/0.3)] transition-all"
                size="lg"
              >
                {isClassifying ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-5 w-5" />
                    Classify Image
                  </>
                )}
              </Button>

              {/* Results */}
              {result && (
                <Card className="p-6 bg-muted/30 border-primary/30 animate-fade-in">
                  <h3 className="text-lg font-semibold mb-3 text-primary flex items-center">
                    <Sparkles className="mr-2 h-5 w-5" />
                    Analysis Results
                  </h3>
                  <div className="prose prose-invert max-w-none">
                    <p className="whitespace-pre-wrap text-foreground">{result}</p>
                  </div>
                </Card>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Classifier;
