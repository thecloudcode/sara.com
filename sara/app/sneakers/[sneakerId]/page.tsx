// app/sneakers/[sneakerId]/page.tsx
import Image from "next/image";
import { notFound } from "next/navigation";
import { sneakersData } from "@/lib/sneaker-data";
import { DynamicPrice } from "@/components/ui/dynamicprice";
import { fetchSneakerPrice } from "@/lib/price-utils";

interface SneakerPageProps {
  params: {
    sneakerId: string;
  };
}

export async function generateStaticParams() {
  return sneakersData.map((sneaker) => ({
    sneakerId: sneaker.id,
  }));
}

export default async function SneakerPage({ params }: SneakerPageProps) {
  // Find the sneaker in static data
  const sneaker = sneakersData.find((s) => s.id === params.sneakerId);

  if (!sneaker) {
    return notFound();
  }

  // Fetch initial price for sneaker
  const initialPrice = await fetchSneakerPrice(params.sneakerId);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="relative aspect-square">
          <Image
            src={sneaker.src}
            alt={sneaker.title}
            fill
            className="object-cover rounded-lg"
            sizes="(max-width: 768px) 100vw, 50vw"
            priority
          />
        </div>
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold">{sneaker.title}</h1>
            <DynamicPrice 
              sneakerId={params.sneakerId} 
              initialPrice={initialPrice} 
            />
          </div>
          {/* Rest of your component remains the same */}
        </div>
      </div>
    </div>
  );
}
