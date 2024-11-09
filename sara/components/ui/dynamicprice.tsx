// components/ui/dynamic-price.tsx

'use client';

import { useEffect, useState } from 'react';
import { fetchSneakerPrice } from '@/lib/price-utils';

interface DynamicPriceProps {
  sneakerId: string;
  initialPrice: number;
}

export function DynamicPrice({ sneakerId, initialPrice }: DynamicPriceProps) {
  const [price, setPrice] = useState(initialPrice);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const updatePrice = async () => {
      try {
        setIsLoading(true);
        const newPrice = await fetchSneakerPrice(sneakerId);
        setPrice(newPrice);
      } catch (error) {
        console.error('Error updating price:', error);
      } finally {
        setIsLoading(false);
      }
    };

    // Update price every 5 minutes
    const interval = setInterval(updatePrice, 5 * 1000);

    return () => clearInterval(interval);
  }, [sneakerId]);

  return (
    <div className="relative">
      <p className="text-xl font-semibold mt-2">
        ${price.toFixed(2)}
        {isLoading && (
          <span className="ml-2 inline-block animate-pulse">
            updating...
          </span>
        )}
      </p>
    </div>
  );
}