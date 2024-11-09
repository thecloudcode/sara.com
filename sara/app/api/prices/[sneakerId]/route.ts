import { NextResponse } from 'next/server';

// This is where you'll integrate your ML model
const getPriceFromML = async (sneakerId: string) => {
  // Replace this with your actual ML model integration
  // This is just a placeholder
  return Math.random() * 100 + 100; // Random price between 100 and 200
};

export async function GET(
  request: Request,
  { params }: { params: { sneakerId: string } }
) {
  try {
    const price = await getPriceFromML(params.sneakerId);
    return NextResponse.json({ price });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch price' }, { status: 500 });
  }
}