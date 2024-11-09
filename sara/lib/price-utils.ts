// lib/price-utils.ts

// lib/price-utils.ts

export async function fetchSneakerPrice(sneakerId: string): Promise<number> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:3000'; // Adjust localhost for your dev setup
  try {
    const response = await fetch(`${baseUrl}/api/prices/${sneakerId}`, {
      next: { revalidate: 300 } // Revalidate every 5 minutes
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.price;
  } catch (error) {
    console.error('Error fetching price:', error);
    return 0; // or handle as needed
  }
}


export async function fetchAllPrices(sneakerIds: string[]): Promise<Record<string, number>> {
  try {
    const response = await fetch('/api/prices', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ids: sneakerIds }),
      next: { revalidate: 300 } // Revalidate every 5 minutes
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.prices;
  } catch (error) {
    console.error('Error fetching prices:', error);
    // Return empty object or throw error based on your needs
    return {}; // or throw error
  }
}