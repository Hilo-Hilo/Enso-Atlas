/**
 * Proxy for model-specific heatmap images - avoids CORS issues
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://100.111.126.23:8003";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string; modelId: string }> }
) {
  const { slideId, modelId } = await params;
  const { searchParams } = new URL(request.url);
  const level = searchParams.get("level");
  
  try {
    let backendUrl = `${BACKEND_URL}/api/heatmap/${encodeURIComponent(slideId)}/${encodeURIComponent(modelId)}`;
    if (level) {
      backendUrl += `?level=${level}`;
    }
    
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
    const contentType = response.headers.get("Content-Type") || "image/png";
    
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (error) {
    console.error(`[Heatmap Proxy] Error fetching heatmap for ${slideId}/${modelId}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch heatmap from backend" },
      { status: 502 }
    );
  }
}
