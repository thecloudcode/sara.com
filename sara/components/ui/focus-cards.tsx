// components/ui/focus-cards.tsx

"use client";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import Link from "next/link";

export const Card = React.memo(
  ({
    card,
    index,
    hovered,
    setHovered,
  }: {
    card: {
      id: string;
      title: string;
      src: string;
      des: string;
    };
    index: number;
    hovered: number | null;
    setHovered: React.Dispatch<React.SetStateAction<number | null>>;
  }) => {
    const [isMounted, setIsMounted] = useState(false);

    useEffect(() => {
      setIsMounted(true);
    }, []);

    return (
      <Link href={`/sneakers/${card.id}`}>
        <div
          onMouseEnter={() => setHovered(index)}
          onMouseLeave={() => setHovered(null)}
          className={cn(
            "rounded-lg relative bg-gray-100 dark:bg-neutral-900 overflow-hidden h-60 md:h-96 w-full transition-all duration-300 ease-out cursor-pointer",
            isMounted && hovered !== null && hovered !== index && "blur-sm scale-[0.98]"
          )}
        >
          <Image
            src={card.src}
            alt={card.title}
            fill
            className="object-cover absolute inset-0"
          />
          <div
            className={cn(
              "absolute inset-0 bg-black/50 flex items-end py-8 px-4 transition-opacity duration-300",
              isMounted && hovered === index ? "opacity-100" : "opacity-0"
            )}
          >
            <div>
              <div className="text-xl md:text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-200">
                {card.title}
              </div>
              <div className="text-sm text-neutral-200 mt-1">
                {card.des}
              </div>
            </div>
          </div>
        </div>
      </Link>
    );
  }
);
Card.displayName = "Card";

type Card = {
  id: string;
  title: string;
  src: string;
  des: string;
};

export function FocusCards({ cards }: { cards: Card[] }) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-10 max-w-5xl mx-auto md:px-8 w-full">
      {cards.map((card, index) => (
        <Card
          key={card.title}
          card={card}
          index={index}
          hovered={hovered}
          setHovered={setHovered}
        />
      ))}
    </div>
  );
}