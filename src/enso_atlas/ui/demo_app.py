"""
Demo UI for Enso Atlas - works with pre-computed embeddings.

This is a simplified version that demonstrates the core functionality
without requiring actual WSI files or the full model pipeline.

Key Features:
- Predictions with confidence scores
- Attention heatmaps overlaid on slide thumbnail
- Evidence patch gallery (top-K by attention)
- Similar patch retrieval (FAISS)
- Structured report generation
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
    
    # Load model
    config = MILConfig(input_dim=384, hidden_dim=128)
    classifier = CLAMClassifier(config)
    
    if model_path.exists():
        classifier.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning("No trained model found, using random weights")
    
    # Setup evidence generator with FAISS index
    evidence_config = EvidenceConfig()
    evidence_gen = EvidenceGenerator(evidence_config)
    
    # Build reference index from all demo slides for similarity search
    all_embeddings = []
    all_metadata = []
    
    available_slides = []
    if embeddings_dir.exists():
        for f in sorted(embeddings_dir.glob("*.npy")):
            if not f.name.endswith("_coords.npy"):
                slide_id = f.stem
                available_slides.append(slide_id)
                embs = np.load(f)
                all_embeddings.append(embs)
                all_metadata.append({
                    "slide_id": slide_id,
                    "n_patches": len(embs)
                })
    
    # Build FAISS index for similarity search
    if all_embeddings:
        evidence_gen.build_reference_index(all_embeddings, all_metadata)
        logger.info(f"Built FAISS index with {len(available_slides)} slides")
    
    def create_fake_thumbnail(size=(512, 512), seed=None):
        """Create a fake H&E-like thumbnail."""
        if seed is not None:
            np.random.seed(seed)
        arr = np.random.randint(200, 255, (*size, 3), dtype=np.uint8)
        # Add some pink/purple tones typical of H&E
        arr[:, :, 0] = np.clip(arr[:, :, 0] + 30, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - 20, 0, 255)
        
        # Add some tissue-like regions
        for _ in range(np.random.randint(3, 8)):
            cx, cy = np.random.randint(50, size[0]-50, 2)
            r = np.random.randint(30, 100)
            y, x = np.ogrid[:size[0], :size[1]]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            color = np.random.randint(150, 200, 3, dtype=np.uint8)
            arr[mask] = color
        
        return Image.fromarray(arr)
    
    def create_patch_image(seed=None):
        """Create a fake H&E patch."""
        if seed is not None:
            np.random.seed(seed)
        patch = np.random.randint(180, 240, (64, 64, 3), dtype=np.uint8)
        patch[:, :, 0] = np.clip(patch[:, :, 0] + 20, 0, 255)
        return Image.fromarray(patch)
    
    def analyze_slide(slide_name: str):
        """Analyze a pre-computed slide with full explainability."""
        if not slide_name:
            return None, "Please select a slide", "", [], "", []
        
        emb_path = embeddings_dir / f"{slide_name}.npy"
        coord_path = embeddings_dir / f"{slide_name}_coords.npy"
        
        if not emb_path.exists():
            return None, f"Embeddings not found for {slide_name}", "", [], "", []
        
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
        slide_dims = (50000, 50000)
        heatmap = evidence_gen.create_heatmap(attention, coords, slide_dims, (512, 512))
        
        # Create thumbnail with heatmap overlay
        seed = hash(slide_name) % 10000
        thumbnail = create_fake_thumbnail(seed=seed)
        thumbnail_arr = np.array(thumbnail)
        
        # Blend heatmap
        heatmap_rgb = heatmap[:, :, :3]
        heatmap_alpha = heatmap[:, :, 3:4] / 255.0
        heatmap_rgb = cv2.resize(heatmap_rgb, (512, 512))
        heatmap_alpha = cv2.resize(heatmap_alpha, (512, 512))[:, :, np.newaxis]
        
        blended = (thumbnail_arr * (1 - heatmap_alpha * 0.7) + heatmap_rgb * heatmap_alpha * 0.7).astype(np.uint8)
        
        # Results text
        result_text = f"""## Prediction Results

| Metric | Value |
|--------|-------|
| **Slide** | {slide_name} |
| **Prediction** | {label} |
| **Score** | {score:.3f} |
| **Confidence** | {confidence:.1%} |
| **Patches analyzed** | {len(embeddings)} |

---
*Attention heatmap shows regions driving the prediction. Red = high attention.*
"""
        
        # Top evidence patches
        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]
        
        evidence_text = "## Top Evidence Patches\n\n"
        evidence_text += "These patches have the highest attention weights and most influence the prediction.\n\n"
        
        patch_images = []
        for i, idx in enumerate(top_indices):
            evidence_text += f"**Patch {i+1}** (attention: {attention[idx]:.4f})\n"
            evidence_text += f"- Location: ({coords[idx][0]}, {coords[idx][1]})\n\n"
            patch_images.append(create_patch_image(seed=seed+idx))
        
        # Similar cases from reference cohort
        similar_cases = evidence_gen.find_similar(embeddings, attention, k=5, top_patches=3)
        
        similar_text = "## Similar Cases\n\n"
        similar_text += "Cases from the reference cohort with similar morphological patterns.\n\n"
        
        if similar_cases:
            seen_slides = set()
            for s in similar_cases[:10]:
                meta = s.get('metadata', {})
                sid = meta.get('slide_id', 'unknown')
                if sid not in seen_slides and sid != slide_name:
                    seen_slides.add(sid)
                    similar_text += f"- **{sid}** (distance: {s['distance']:.3f})\n"
                    if len(seen_slides) >= 5:
                        break
        else:
            similar_text += "*No similar cases found in reference cohort.*\n"
        
        # Similar patch images
        similar_images = []
        for i, s in enumerate(similar_cases[:6]):
            meta = s.get('metadata', {})
            sid = meta.get('slide_id', 'unknown')
            similar_images.append(create_patch_image(seed=hash(sid+str(i)) % 10000))
        
        return (
            Image.fromarray(blended),
            result_text,
            evidence_text,
            patch_images,
            similar_text,
            similar_images
        )
    
    def generate_report(slide_name: str):
        """Generate a structured report."""
        if not slide_name:
            return "Please analyze a slide first."
        
        emb_path = embeddings_dir / f"{slide_name}.npy"
        if not emb_path.exists():
            return "Slide not found."
        
        embeddings = np.load(emb_path)
        score, attention = classifier.predict(embeddings)
        label = "responder" if score > 0.5 else "non-responder"
        
        report = {
            "case_id": slide_name,
            "task": "Bevacizumab treatment response prediction",
            "model_output": {
                "label": label,
                "probability": float(score),
                "calibration_note": "Model probability. Requires external validation."
            },
            "evidence": [
                {
                    "patch_id": f"patch_{i}",
                    "attention_weight": float(attention[idx]),
                    "significance": "High attention region"
                }
                for i, idx in enumerate(np.argsort(attention)[-5:][::-1])
            ],
            "limitations": [
                "Demo mode with synthetic data",
                "Not clinically validated",
                "Requires pathologist review"
            ],
            "safety_statement": "This is a research tool. All findings require validation by qualified pathologists."
        }
        
        return f"""## Structured Report

```json
{json.dumps(report, indent=2)}
```

---

### Summary

**Case:** {slide_name}  
**Prediction:** {label.upper()}  
**Score:** {score:.3f}

**Key Evidence:**
- Analyzed {len(embeddings)} tissue patches
- Top {min(5, len(attention))} patches show consistent morphological patterns

**Limitations:**
- This is a demonstration with synthetic data
- Real clinical use requires validation with actual WSI data
- All findings must be reviewed by qualified pathologists

**Safety Statement:**  
This tool is for research purposes only. Do not use for clinical decision-making without proper validation.
"""
    
    # Build interface
    with gr.Blocks(
        title="Enso Atlas - Pathology Evidence Engine",
    ) as app:
        
        gr.Markdown("""
# 🔬 Enso Atlas

**On-Prem Pathology Evidence Engine for Treatment-Response Insight**

This demo shows the core explainability features:
- **Attention Heatmaps** - visualize which regions drive predictions
- **Evidence Patches** - top contributing tissue regions
- **Similar Cases** - reference cohort comparison
- **Structured Reports** - tumor board-ready summaries
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                slide_dropdown = gr.Dropdown(
                    choices=available_slides,
                    label="Select Slide",
                    value=available_slides[0] if available_slides else None,
                )
                
                analyze_btn = gr.Button("🔍 Analyze Slide", variant="primary", size="lg")
                report_btn = gr.Button("📋 Generate Report", variant="secondary")
                
                result_text = gr.Markdown(label="Prediction Results")
            
            with gr.Column(scale=2):
                heatmap_image = gr.Image(label="Attention Heatmap", type="pil", height=400)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Evidence Patches")
                evidence_text = gr.Markdown()
                evidence_gallery = gr.Gallery(
                    label="Top Evidence Patches (by attention)",
                    columns=4,
                    rows=2,
                    height=150,
                )
            
            with gr.Column():
                gr.Markdown("### Similar Cases")
                similar_text = gr.Markdown()
                similar_gallery = gr.Gallery(
                    label="Similar patches from reference cohort",
                    columns=3,
                    rows=2,
                    height=150,
                )
        
        with gr.Row():
            report_output = gr.Markdown(label="Structured Report")
        
        gr.Markdown("""
---
⚠️ **Research Tool Notice:** This is a demonstration using synthetic data. 
In production:
- Real WSI files are processed with Path Foundation embeddings
- MedGemma generates detailed tumor board reports
- All processing runs locally (no PHI leaves the network)
- Results require validation by qualified pathologists

[MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
        """)
        
        # Wire up events
        analyze_btn.click(
            fn=analyze_slide,
            inputs=[slide_dropdown],
            outputs=[
                heatmap_image,
                result_text,
                evidence_text,
                evidence_gallery,
                similar_text,
                similar_gallery,
            ],
        )
        
        report_btn.click(
            fn=generate_report,
            inputs=[slide_dropdown],
            outputs=[report_output],
        )
        
        # Auto-analyze first slide on load
        if available_slides:
            app.load(
                fn=analyze_slide,
                inputs=[slide_dropdown],
                outputs=[
                    heatmap_image,
                    result_text,
                    evidence_text,
                    evidence_gallery,
                    similar_text,
                    similar_gallery,
                ],
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
