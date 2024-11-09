import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { ids } = await request.json();
    const prices: Record<string, number> = {};
    
    // Fetch prices for all sneakers in parallel
    await Promise.all(
      ids.map(async (id: string) => {
        // Replace with your ML model integration
        prices[id] = Math.random() * 100 + 100;
      })
    );
    
    return NextResponse.json({ prices });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch prices' }, { status: 500 });
  }
}