/**
 * Proxy for DZI descriptor - avoids CORS issues by routing through same origin
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8003";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string }> }
) {
  const { slideId } = await params;
  
  try {
    const backendUrl = `${BACKEND_URL}/api/slides/${encodeURIComponent(slideId)}/dzi`;
    
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Accept": "application/xml",
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend returned ${response.status}` },
        { status: response.status }
      );
    }

    const xmlContent = await response.text();
    
    return new NextResponse(xmlContent, {
      status: 200,
      headers: {
        "Content-Type": "application/xml",
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (error) {
    console.error(`[DZI Proxy] Error fetching DZI for ${slideId}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch DZI descriptor from backend" },
      { status: 502 }
    );
  }
}

// HEAD handler for DZI availability pre-check (used by WSIViewer to avoid OSD error flash)
export async function HEAD(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string }> }
) {
  const { slideId } = await params;

  try {
    const backendUrl = `${BACKEND_URL}/api/slides/${encodeURIComponent(slideId)}/dzi`;
    const response = await fetch(backendUrl, { method: "GET" });

    return new NextResponse(null, {
      status: response.ok ? 200 : response.status,
      headers: {
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch {
    return new NextResponse(null, { status: 502 });
  }
}
