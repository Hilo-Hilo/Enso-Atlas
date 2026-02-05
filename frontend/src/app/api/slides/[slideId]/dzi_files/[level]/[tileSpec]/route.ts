/**
 * Proxy for DZI tiles - avoids CORS issues by routing through same origin
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://100.111.126.23:8003";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string; level: string; tileSpec: string }> }
) {
  const { slideId, level, tileSpec } = await params;
  
  try {
    const backendUrl = `${BACKEND_URL}/api/slides/${encodeURIComponent(slideId)}/dzi_files/${level}/${tileSpec}`;
    
    const response = await fetch(backendUrl, {
      method: "GET",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend returned ${response.status}` },
        { status: response.status }
      );
    }

    const imageBuffer = await response.arrayBuffer();
    
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch (error) {
    console.error(`[Tile Proxy] Error fetching tile ${level}/${tileSpec} for ${slideId}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch tile from backend" },
      { status: 502 }
    );
  }
}
