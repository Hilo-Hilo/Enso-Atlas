"""
Demo UI for Enso Atlas - works with pre-computed embeddings.

This is a simplified version that demonstrates the core functionality
without requiring actual WSI files or the full model pipeline.
"""

from pathlib import Path
from typing import Optional, List
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)


def create_demo_app(
    embeddings_dir: Path = Path("data/demo/embeddings"),
    model_path: Path = Path("models/demo_clam.pt"),
):
    """
    Create a demo Gradio interface using pre-computed embeddings.
    """
    import gradio as gr
    from PIL import Image
    import cv2
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from enso_atlas.config import MILConfig, EvidenceConfig
    from enso_atlas.mil.clam import CLAMClassifier
    from enso_atlas.evidence.generator import EvidenceGenerator
    
    # Load model (match the config used during training)
    config = MILConfig(input_dim=384, hidden_dim=128)
    classifier = CLAMClassifier(config)
    
    if model_path.exists():
        classifier.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning("No trained model found, using random weights")
    
    # Setup evidence generator
    evidence_config = EvidenceConfig()
    evidence_gen = EvidenceGenerator(evidence_config)
    
    # Get available demo slides
    available_slides = []
    if embeddings_dir.exists():
        for f in embeddings_dir.glob("*.npy"):
            if not f.name.endswith("_coords.npy"):
                available_slides.append(f.stem)
    
    def create_fake_thumbnail(size=(512, 512)):
        """Create a fake H&E-like thumbnail."""
        arr = np.random.randint(200, 255, (*size, 3), dtype=np.uint8)
        # Add some pink/purple tones typical of H&E
        arr[:, :, 0] = np.clip(arr[:, :, 0] + 30, 0, 255)  # More red
        arr[:, :, 2] = np.clip(arr[:, :, 2] - 20, 0, 255)  # Less blue
        return Image.fromarray(arr)
    
    def analyze_slide(slide_name: str):
        """Analyze a pre-computed slide."""
        if not slide_name:
            return None, "Please select a slide", "", []
        
        emb_path = embeddings_dir / f"{slide_name}.npy"
        coord_path = embeddings_dir / f"{slide_name}_coords.npy"
        
        if not emb_path.exists():
            return None, f"Embeddings not found for {slide_name}", "", []
        
        # Load embeddings
        embeddings = np.load(emb_path)
        
        # Load or generate coordinates
        if coord_path.exists():
            coords = np.load(coord_path)
        else:
            coords = np.random.randint(0, 50000, (len(embeddings), 2))
        
        coords = [tuple(c) for c in coords]
        
        # Run prediction
        score, attention = classifier.predict(embeddings)
        label = "RESPONDER" if score > 0.5 else "NON-RESPONDER"
        confidence = abs(score - 0.5) * 2
        
        # Create heatmap
        slide_dims = (50000, 50000)  # Fake dimensions
        heatmap = evidence_gen.create_heatmap(attention, coords, slide_dims, (512, 512))
        
        # Create fake thumbnail with heatmap overlay
        thumbnail = create_fake_thumbnail()
        thumbnail_arr = np.array(thumbnail)
        
        # Blend heatmap
        heatmap_rgb = heatmap[:, :, :3]
        heatmap_alpha = heatmap[:, :, 3:4] / 255.0
        
        # Resize heatmap to match thumbnail
        heatmap_rgb = cv2.resize(heatmap_rgb, (512, 512))
        heatmap_alpha = cv2.resize(heatmap_alpha, (512, 512))[:, :, np.newaxis]
        
        blended = (thumbnail_arr * (1 - heatmap_alpha * 0.7) + heatmap_rgb * heatmap_alpha * 0.7).astype(np.uint8)
        
        # Format results
        result_text = f"""## Analysis Results

**Slide:** {slide_name}
**Prediction:** {label}
**Score:** {score:.3f}
**Confidence:** {confidence:.1%}

**Patches analyzed:** {len(embeddings)}

---
*This is a demo using pre-computed embeddings. In production, Path Foundation generates these from real WSI patches.*
"""
        
        # Top evidence patches info
        top_k = min(6, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]
        
        evidence_text = "## Top Evidence Patches\n\n"
        for i, idx in enumerate(top_indices):
            evidence_text += f"**Patch {i+1}** (attention: {attention[idx]:.3f})\n"
            evidence_text += f"- Location: ({coords[idx][0]}, {coords[idx][1]})\n\n"
        
        # Create fake patch thumbnails
        patch_images = []
        for idx in top_indices:
            # Random pink/purple patches to simulate H&E
            patch = np.random.randint(180, 240, (64, 64, 3), dtype=np.uint8)
            patch[:, :, 0] = np.clip(patch[:, :, 0] + 20, 0, 255)
            patch_images.append(Image.fromarray(patch))
        
        return Image.fromarray(blended), result_text, evidence_text, patch_images
    
    # Build interface
    with gr.Blocks(
        title="Enso Atlas Demo",
        theme=gr.themes.Soft(),
    ) as app:
        
        gr.Markdown("""
# 🔬 Enso Atlas - Demo Mode

**On-Prem Pathology Evidence Engine for Treatment-Response Insight**

This demo uses pre-computed embeddings to demonstrate the analysis pipeline.
Select a demo slide to see predictions, attention heatmaps, and evidence patches.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                slide_dropdown = gr.Dropdown(
                    choices=available_slides,
                    label="Select Demo Slide",
                    value=available_slides[0] if available_slides else None,
                )
                
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                result_text = gr.Markdown(label="Results")
            
            with gr.Column(scale=2):
                heatmap_image = gr.Image(label="Attention Heatmap", type="pil")
        
        with gr.Row():
            with gr.Column():
                evidence_text = gr.Markdown(label="Evidence")
                
                evidence_gallery = gr.Gallery(
                    label="Top Evidence Patches",
                    columns=6,
                    rows=1,
                    height=100,
                )
        
        gr.Markdown("""
---
⚠️ **Demo Notice:** This is running with synthetic data. In production:
- Real WSI files are uploaded and processed
- Path Foundation generates 384-dim embeddings from H&E patches  
- MedGemma generates structured tumor board reports
- All processing runs locally (no PHI leaves the hospital network)
        """)
        
        analyze_btn.click(
            fn=analyze_slide,
            inputs=[slide_dropdown],
            outputs=[heatmap_image, result_text, evidence_text, evidence_gallery],
        )
        
        # Auto-analyze on load if slides available
        if available_slides:
            app.load(
                fn=analyze_slide,
                inputs=[slide_dropdown],
                outputs=[heatmap_image, result_text, evidence_text, evidence_gallery],
            )
    
    return app


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/demo/embeddings"))
    parser.add_argument("--model-path", type=Path, default=Path("models/demo_clam.pt"))
    
    args = parser.parse_args()
    
    app = create_demo_app(args.embeddings_dir, args.model_path)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
