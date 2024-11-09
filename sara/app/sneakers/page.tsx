// app/sneakers/page.tsx

import { fetchAllPrices } from '@/lib/price-utils';
import { sneakersData } from '@/lib/sneaker-data';
import { FocusCards } from '@/components/ui/focus-cards';

export default async function Home() {
  // Fetch initial prices for all sneakers
  const sneakerIds = sneakersData.map(sneaker => sneaker.id);
  const prices = await fetchAllPrices(sneakerIds);

  // Add prices to sneaker data
  const sneakersWithPrices = sneakersData.map(sneaker => ({
    ...sneaker,
    price: prices[sneaker.id],
  }));

  return (
    <div className="p-10">
      <h1 className="text-4xl md:text-4xl lg:text-6xl font-semibold max-w-7xl mx-auto text-center mt-20 relative z-20 py-6 bg-clip-text text-transparent bg-gradient-to-b from-neutral-800 via-neutral-700 to-neutral-700 dark:from-neutral-800 dark:via-white dark:to-white">
        Have a look at our <br /> immense collection
      </h1>
      <div className="mt-20">
        <FocusCards cards={sneakersWithPrices} />
      </div>
    </div>
  );
}