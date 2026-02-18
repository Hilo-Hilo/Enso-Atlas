/**
 * Proxy for model-specific heatmap images - avoids CORS issues
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8003";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string; modelId: string }> }
) {
  const { slideId, modelId } = await params;
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
    const backendUrl = `${BACKEND_URL}/api/heatmap/${encodeURIComponent(slideId)}/${encodeURIComponent(modelId)}${qs}`;
    
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
    
    const headers: Record<string, string> = {
      "Content-Type": contentType,
      "Cache-Control": alphaPower ? "no-cache" : "public, max-age=3600",
    };
    // Forward coverage headers for heatmap alignment
    for (const h of ["X-Model-Id", "X-Model-Name", "X-Slide-Width", "X-Slide-Height", "X-Coverage-Width", "X-Coverage-Height"]) {
      const val = response.headers.get(h);
      if (val) headers[h] = val;
    }
    headers["Access-Control-Expose-Headers"] = "X-Model-Id, X-Model-Name, X-Slide-Width, X-Slide-Height, X-Coverage-Width, X-Coverage-Height";
    
    return new NextResponse(imageBuffer, { status: 200, headers });
  } catch (error) {
    console.error(`[Heatmap Proxy] Error fetching heatmap for ${slideId}/${modelId}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch heatmap from backend" },
      { status: 502 }
    );
  }
}
