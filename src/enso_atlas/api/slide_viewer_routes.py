"""Slide viewer support routes that sit above WSI tile serving."""

import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image

logger = logging.getLogger(__name__)


def create_slide_viewer_router(
    *,
    require_project: Callable[[str | None], Any],
    get_slide_and_dz: Callable[..., Any],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_slide_path: Callable[..., Path | None],
    infer_patch_size_from_coords: Callable[..., int],
    normalize_coords_to_level0: Callable[..., tuple[Any, int]],
    embeddings_dir_provider: Callable[[], Path],
    thumbnail_cache_dir: Path,
) -> APIRouter:
    router = APIRouter()

    @router.api_route("/api/slides/{slide_id}/dzi", methods=["GET", "HEAD"])
    async def get_dzi_descriptor(
        request: Request,
        slide_id: str,
        project_id: str | None = Query(
            None, description="Optional project id to resolve project-specific WSI paths"
        ),
    ):
        """Get or HEAD-check a DZI descriptor with optional project scoping."""
        require_project(project_id)
        result = get_slide_and_dz(slide_id, project_id=project_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "WSI_NOT_FOUND",
                    "message": f"WSI file not found for slide {slide_id}",
                    "slide_id": slide_id,
                    "has_wsi": False,
                },
            )

        if request.method == "HEAD":
            return Response(
                status_code=200,
                headers={
                    "Content-Type": "application/xml",
                    "Cache-Control": "public, max-age=3600",
                },
            )

        _slide, dz = result

        dzi_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpeg"
  Overlap="1"
  TileSize="254">
  <Size Width="{dz.level_dimensions[-1][0]}" Height="{dz.level_dimensions[-1][1]}"/>
</Image>'''

        return Response(
            content=dzi_xml,
            media_type="application/xml",
            headers={"Content-Disposition": f"inline; filename={slide_id}.dzi"},
        )

    @router.get("/api/slides/{slide_id}/dzi_files/{level}/{tile_spec}")
    async def get_dzi_tile(
        slide_id: str,
        level: int,
        tile_spec: str,
        project_id: str | None = Query(
            None, description="Optional project id to resolve project-specific WSI paths"
        ),
    ):
        """Serve one DZI tile image from a project-aware WSI path."""
        require_project(project_id)
        result = get_slide_and_dz(slide_id, project_id=project_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"WSI file not found for slide {slide_id}",
            )

        _slide, dz = result

        try:
            tile_name = tile_spec.rsplit(".", 1)[0]
            col, row = map(int, tile_name.split("_"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tile specification")

        if level < 0 or level >= dz.level_count:
            raise HTTPException(status_code=404, detail="Invalid zoom level")

        level_tiles = dz.level_tiles[level]
        if col < 0 or col >= level_tiles[0] or row < 0 or row >= level_tiles[1]:
            raise HTTPException(status_code=404, detail="Tile coordinates out of bounds")

        try:
            tile = dz.get_tile(level, (col, row))

            buf = io.BytesIO()
            tile.save(buf, format="JPEG", quality=85)
            buf.seek(0)

            return StreamingResponse(
                buf,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Content-Disposition": f"inline; filename={level}_{col}_{row}.jpeg",
                },
            )
        except Exception as e:
            logger.error("Failed to get tile %s/%s_%s for %s: %s", level, col, row, slide_id, e)
            raise HTTPException(status_code=500, detail=f"Failed to generate tile: {e}")

    @router.get("/api/slides/{slide_id}/thumbnail")
    async def get_slide_thumbnail(
        slide_id: str,
        size: int = 256,
        project_id: str | None = Query(
            None, description="Optional project id to resolve project-specific WSI paths"
        ),
    ):
        """Get a whole-slide thumbnail with optional project scoping."""
        size = max(64, min(size, 1024))
        require_project(project_id)

        cache_prefix = project_id if project_id else "global"
        cache_path = thumbnail_cache_dir / f"{cache_prefix}_{slide_id}_{size}.jpg"
        if cache_path.exists():
            return FileResponse(
                cache_path,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "X-WSI-Available": "true",
                },
            )

        result = get_slide_and_dz(slide_id, project_id=project_id)
        if result is None:
            from PIL import ImageDraw

            img = Image.new("RGB", (size, size), (237, 242, 247))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, size - 1, size - 1], outline=(203, 213, 225), width=2)

            try:
                from PIL import ImageFont

                font = ImageFont.load_default()
            except Exception:
                font = None

            title = "Embeddings"
            subtitle = "WSI unavailable"
            tb = draw.textbbox((0, 0), title, font=font)
            sb = draw.textbbox((0, 0), subtitle, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            sw = sb[2] - sb[0]
            sh = sb[3] - sb[1]
            draw.text(
                ((size - tw) / 2, (size - th - sh - 6) / 2), title, fill=(71, 85, 105), font=font
            )
            draw.text(
                ((size - sw) / 2, (size - th - sh - 6) / 2 + th + 6),
                subtitle,
                fill=(100, 116, 139),
                font=font,
            )

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "X-WSI-Available": "false",
                    "X-Thumbnail-Fallback": "embeddings-only",
                },
            )

        slide, _dz = result

        try:
            thumb = slide.get_thumbnail((size, size))
            thumb.save(cache_path, format="JPEG", quality=90)
            logger.info("Cached thumbnail for %s at size %s", slide_id, size)

            return FileResponse(
                cache_path,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "X-WSI-Available": "true",
                },
            )
        except Exception as e:
            logger.error("Failed to get thumbnail for %s: %s", slide_id, e)
            raise HTTPException(status_code=500, detail=f"Failed to generate thumbnail: {e}")

    @router.get("/api/slides/{slide_id}/patches/{patch_id}")
    async def get_patch_image(
        slide_id: str,
        patch_id: str,
        size: int = 224,
        project_id: str | None = Query(
            None, description="Optional project id to resolve project-specific embeddings/WSI paths"
        ),
        x: int | None = Query(
            None, description="Optional explicit level-0 x coordinate for direct patch extraction"
        ),
        y: int | None = Query(
            None, description="Optional explicit level-0 y coordinate for direct patch extraction"
        ),
        patch_size: int | None = Query(
            None, description="Optional level-0 patch span in pixels (for low-mag semantic patches)"
        ),
    ):
        """Get a patch image thumbnail for semantic search and evidence panels."""
        require_project(project_id)
        size = max(64, min(int(size), 1024))

        try:
            if patch_id.startswith("patch_"):
                patch_index = int(patch_id.replace("patch_", ""))
            else:
                patch_index = int(patch_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid patch ID format: {patch_id}")

        project_requested = project_id is not None
        patch_embeddings_dir = resolve_project_embeddings_dir(
            project_id,
            require_exists=project_requested,
        )

        emb_path = patch_embeddings_dir / f"{slide_id}.npy"
        coord_path = patch_embeddings_dir / f"{slide_id}_coords.npy"
        siglip_coords_path = (
            patch_embeddings_dir / "medsiglip_cache" / f"{slide_id}_siglip_coords.npy"
        )

        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        siglip_coords = None
        if siglip_coords_path.exists():
            try:
                siglip_coords = np.load(siglip_coords_path)
                slide_dims = None
                slide_path = resolve_slide_path(slide_id, project_id=project_id)
                if slide_path is not None and slide_path.exists():
                    try:
                        import openslide

                        with openslide.OpenSlide(str(slide_path)) as slide_obj:
                            slide_dims = (
                                int(slide_obj.dimensions[0]),
                                int(slide_obj.dimensions[1]),
                            )
                    except Exception as e:
                        logger.debug(
                            "Could not read slide dimensions for patch extraction normalization: %s",
                            e,
                        )
                siglip_coords, _ = normalize_coords_to_level0(
                    siglip_coords,
                    slide_dims=slide_dims,
                    patch_size=224,
                )
            except Exception as e:
                logger.warning("Failed to load MedSigLIP coordinates for %s: %s", slide_id, e)
                siglip_coords = None

        explicit_coords = x is not None and y is not None
        if not explicit_coords:
            max_count = 0
            if emb_path.exists():
                try:
                    max_count = max(max_count, len(np.load(emb_path)))
                except Exception as e:
                    logger.warning("Could not load embedding count for %s: %s", slide_id, e)
            if coords is not None:
                max_count = max(max_count, len(coords))
            if siglip_coords is not None:
                max_count = max(max_count, len(siglip_coords))

            if max_count == 0:
                raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

            if patch_index < 0 or patch_index >= max_count:
                raise HTTPException(
                    status_code=404,
                    detail=f"Patch index {patch_index} out of range (0-{max_count - 1})",
                )

        pf_patch_size = infer_patch_size_from_coords(coords, default_patch_size=224)
        siglip_patch_size = infer_patch_size_from_coords(siglip_coords, default_patch_size=224)
        requested_patch_size = int(patch_size) if patch_size is not None else None

        x0: int | None = None
        y0: int | None = None
        effective_patch_size = requested_patch_size or 224

        if explicit_coords:
            if x is None or y is None:
                raise HTTPException(status_code=400, detail="Both x and y coordinates are required")
            x0 = int(x)
            y0 = int(y)
            effective_patch_size = requested_patch_size or effective_patch_size
        else:
            if siglip_coords is not None and patch_index < len(siglip_coords):
                x0 = int(siglip_coords[patch_index][0])
                y0 = int(siglip_coords[patch_index][1])
                effective_patch_size = requested_patch_size or siglip_patch_size
            elif coords is not None and patch_index < len(coords):
                x0 = int(coords[patch_index][0])
                y0 = int(coords[patch_index][1])
                effective_patch_size = requested_patch_size or pf_patch_size

        effective_patch_size = max(64, min(int(effective_patch_size), 4096))

        result = get_slide_and_dz(slide_id, project_id=project_id)
        if result is not None and x0 is not None and y0 is not None:
            slide, _dz = result
            try:
                region = slide.read_region(
                    (x0, y0), 0, (effective_patch_size, effective_patch_size)
                )

                if region.mode == "RGBA":
                    background = Image.new("RGB", region.size, (255, 255, 255))
                    background.paste(region, mask=region.split()[3])
                    region = background
                elif region.mode != "RGB":
                    region = region.convert("RGB")

                if region.size != (size, size):
                    try:
                        resample = Image.Resampling.BILINEAR
                    except AttributeError:
                        resample = Image.BILINEAR
                    region = region.resize((size, size), resample=resample)

                buf = io.BytesIO()
                region.save(buf, format="JPEG", quality=85)
                buf.seek(0)

                return StreamingResponse(
                    buf,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                        "Content-Disposition": f"inline; filename={slide_id}_{patch_id}.jpeg",
                    },
                )
            except Exception as e:
                logger.warning("Failed to extract patch from WSI: %s", e)

        import colorsys

        hue = (patch_index * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.3, 0.9)
        color = (int(r * 255), int(g * 255), int(b * 255))

        img = Image.new("RGB", (size, size), color)

        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(img)

            text = f"Patch {patch_index}"
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x_pos = (size - text_width) // 2
            y_pos = (size - text_height) // 2

            draw.text((x_pos, y_pos), text, fill=(60, 60, 60), font=font)

            if x0 is not None and y0 is not None:
                coord_text = f"({x0}, {y0})"
                coord_bbox = draw.textbbox((0, 0), coord_text, font=font)
                coord_width = coord_bbox[2] - coord_bbox[0]
                draw.text(
                    ((size - coord_width) // 2, y_pos + text_height + 5),
                    coord_text,
                    fill=(80, 80, 80),
                    font=font,
                )
        except Exception as e:
            logger.debug("Could not add text to placeholder: %s", e)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f"inline; filename={slide_id}_{patch_id}.jpeg",
            },
        )

    @router.get("/api/slides/{slide_id}/info")
    async def get_slide_info(slide_id: str):
        """Get detailed information about a WSI file."""
        result = get_slide_and_dz(slide_id)
        embeddings_dir = embeddings_dir_provider()
        if result is None:
            emb_path = embeddings_dir / f"{slide_id}.npy"
            if emb_path.exists():
                embeddings = np.load(emb_path)
                return {
                    "slide_id": slide_id,
                    "has_wsi": False,
                    "has_embeddings": True,
                    "num_patches": len(embeddings),
                }
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")

        slide, dz = result

        return {
            "slide_id": slide_id,
            "has_wsi": True,
            "dimensions": {
                "width": slide.dimensions[0],
                "height": slide.dimensions[1],
            },
            "level_count": slide.level_count,
            "level_dimensions": [list(d) for d in slide.level_dimensions],
            "properties": dict(slide.properties) if hasattr(slide, "properties") else {},
            "dzi": {
                "tile_size": 254,
                "overlap": 1,
                "level_count": dz.level_count,
            },
        }

    return router
