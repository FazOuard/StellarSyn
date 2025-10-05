"use client";
import { AnimatedTestimonials } from "@/components/ui/animated-testimonials";
import { SparklesCore } from "@/components/ui/sparkles";
import { Mail, Youtube, Github, Linkedin } from "lucide-react";

import R1 from "@/assets/R1.png";
import Faz from "@/assets/faz.png";

const informations = [
    {
        quote:
            "A data Engineer passionate about data science and a space lover.", //change this
        name: "Fatima Zahra Ouardirhi Faz", //change this
        designation: "Data Engineer",
        src: Faz, //change this
    },
    {
        quote:
            "An Engineer excited for the opportunities that AI brings to the table. The future is here, and it's incredibly exciting.",
        name: "Amine Raouane R1",
        designation: "AI Engineer",
        src: R1,
    }
];

const AboutUs = () => {
    return (
        <div className="flex-1 flex flex-row items-center justify-center w-full h-full bg-black relative overflow-hidden">
            <div className="absolute inset-0 w-full h-full pointer-events-none">
                <SparklesCore
                    id="tsparticles-section"
                    background="transparent"
                    minSize={0.6}
                    maxSize={1.4}
                    particleDensity={100}
                    className="w-full h-full"
                    particleColor="#FFFFFF"
                />
            </div>

            <div className="h-full w-[55%] flex items-center justify-center relative z-10">
                <div className="w-4/5">
                    <AnimatedTestimonials testimonials={informations} />
                </div>
            </div>

            <div className="bg-transparent h-full flex-1 flex items-center justify-center relative z-10">
                <div className="w-4/5 bg-white/5 backdrop-blur-md rounded-2xl p-8 border-none outline-none shadow-xl text-white space-y-6">
                    <h2 className="text-3xl font-semibold text-center bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
                        Our project is open-source!
                    </h2>

                   

                    <div className="flex flex-col space-y-4 text-center">
                        
                        <a
                            href="https://github.com/FazOuard/StellarSyn"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center justify-center gap-2 hover:text-violet-400 transition-colors"
                        >
                            <Github className="w-5 h-5" />
                            GitHub Repository
                        </a>
                    </div>
                </div>
            </div>


        </div>


    );
}

export default AboutUs;
