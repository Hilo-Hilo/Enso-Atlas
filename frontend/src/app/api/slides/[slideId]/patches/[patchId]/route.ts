/**
 * Proxy for patch images - avoids CORS issues
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://100.111.126.23:8003";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string; patchId: string }> }
) {
  const { slideId, patchId } = await params;
  
  try {
    const backendUrl = `${BACKEND_URL}/api/slides/${encodeURIComponent(slideId)}/patches/${encodeURIComponent(patchId)}`;
    
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
    const contentType = response.headers.get("Content-Type") || "image/jpeg";
    
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch (error) {
    console.error(`[Patch Proxy] Error fetching patch ${patchId} for ${slideId}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch patch from backend" },
      { status: 502 }
    );
  }
}
