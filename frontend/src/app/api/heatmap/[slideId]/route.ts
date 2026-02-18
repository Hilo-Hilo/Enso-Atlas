/**
 * Proxy for heatmap images - avoids CORS issues
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8003";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string }> }
) {
  const { slideId } = await params;
  const { searchParams } = new URL(request.url);

  try {
    const backendParams = new URLSearchParams();
    const level = searchParams.get("level");
    const alphaPower = searchParams.get("alpha_power");
    const projectId = searchParams.get("project_id");
    if (level) backendParams.set("level", level);
    if (alphaPower) backendParams.set("alpha_power", alphaPower);
    if (projectId) backendParams.set("project_id", projectId);

    const qs = backendParams.toString() ? `?${backendParams.toString()}` : "";
    const backendUrl = `${BACKEND_URL}/api/heatmap/${encodeURIComponent(slideId)}${qs}`;
    
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
    console.error(`[Heatmap Proxy] Error fetching heatmap for ${slideId}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch heatmap from backend" },
      { status: 502 }
    );
  }
}
