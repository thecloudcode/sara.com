import Image from "next/image";
import { BackgroundLines } from "@/components/ui/background-lines";
import { AnimatedTooltip } from "@/components/ui/animated-tooltip";
import { AppleCardsCarouselDemo } from "./components/hero";
import React from "react";
import { FeaturesSectionDemo } from "./components/pagedes";

const people = [
  {
    id: 1,
    name: "Badal Prasad Singh",
    designation: "Software Engineer",
    image: "https://avatars.githubusercontent.com/u/114615639?v=4",
  },
  {
    id: 2,
    name: "Mehakpreet Kaur",
    designation: "Data Scientist",
    image: "https://avatars.githubusercontent.com/u/114038404?v=4",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen">
      <div className="relative w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <BackgroundLines className="flex flex-col items-center justify-center w-full py-8 sm:py-12 lg:py-16">
          <div className="space-y-4 sm:space-y-6 lg:space-y-8 text-center">
            
            <h2 className="bg-clip-text text-transparent bg-gradient-to-b from-neutral-900 to-neutral-700 dark:from-neutral-600 dark:to-white text-4xl sm:text-5xl lg:text-7xl font-sans font-bold tracking-tight relative z-20 mt-20">
              sara.ai
            </h2>
            
            <p className="text-base sm:text-lg text-neutral-700 dark:text-neutral-400 mx-auto px-4 uppercase tracking-widest w-full">
              The only e-commerce website with Gray-Matter
            </p>
            
            <div className="flex flex-row items-center justify-center w-full mt-6 sm:mt-8">
              {/* <AnimatedTooltip items={people} /> */}
            </div>
          </div>
        </BackgroundLines>

        {/* Carousel section with proper padding */}
        <div className="w-full py-8 sm:py-12">
          <AppleCardsCarouselDemo />
        </div>
      </div>
    </main>
  );
}