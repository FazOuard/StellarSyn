"use client";
import { motion } from "motion/react";
import { Link } from "react-router-dom";
import { ImagesSlider } from "@/components/ui/images-slider";
import img1 from "@/assets/bh1.png";
import img2 from "@/assets/bh2.png";
import img3 from "@/assets/bh3.png";

const Home = () => {
  const images = [
    img1,
    img2,
    img3
  ];
  return (
    <ImagesSlider className="flex-1" images={images}>
      <motion.div
        initial={{
          opacity: 0,
          y: -80,
        }}
        animate={{
          opacity: 1,
          y: 0,
        }}
        transition={{
          duration: 1.6,
        }}
        className="z-50 flex flex-col justify-center items-center"
      >
        <motion.p className="font-bold text-xl md:text-6xl text-center bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 py-4">
          Gravity binds the cosmos <br /> curiosity binds intelligence
        </motion.p>
        <button className="px-4 py-2 backdrop-blur-sm border bg-emerald-300/10 border-emerald-500/20 text-white mx-auto text-center rounded-full relative mt-4">
          <Link to="/choose">Try now →</Link>
          <div className="absolute inset-x-0  h-px -bottom-px bg-gradient-to-r w-3/4 mx-auto from-transparent via-emerald-500 to-transparent" />
        </button>
      </motion.div>
    </ImagesSlider>
  );
}

export default Home;
