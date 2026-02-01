"""
Gradio UI for Enso Atlas.

Provides an interactive interface for:
- WSI upload and analysis
- Heatmap visualization with zoom
- Evidence patch gallery
- Similar case retrieval
- Report generation and export
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import json
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None):
    """
    Create the Gradio interface for Enso Atlas.

    Args:
        config_path: Optional path to config file

    Returns:
        Gradio Blocks app
    """
    import gradio as gr
    from PIL import Image

    from ..core import EnsoAtlas
    from ..config import AtlasConfig
    from ..evidence.generator import create_overlay_image

    # Load configuration
    if config_path:
        config = AtlasConfig.from_yaml(config_path)
    else:
        config = AtlasConfig()

    # Initialize Atlas (lazy loading of models)
    atlas = EnsoAtlas(config)

    # State storage
    current_result = {"value": None}

    def analyze_slide(slide_file, generate_report: bool = True):
        """Analyze uploaded slide."""
        if slide_file is None:
            return None, None, "Please upload a slide file.", None, None

        try:
            # Get file path
            if hasattr(slide_file, "name"):
                slide_path = slide_file.name
            else:
                slide_path = slide_file

            # Run analysis
            result = atlas.analyze(
                slide_path,
                generate_report=generate_report,
            )

            # Store result
            current_result["value"] = result

            # Get thumbnail
            thumbnail = atlas.wsi_processor.get_thumbnail()

            # Create overlay
            overlay = create_overlay_image(thumbnail, result.heatmap)

            # Format prediction
            pred_text = f"""
## Prediction Results

**Classification:** {result.label.upper()}
**Probability Score:** {result.score:.3f}
**Confidence:** {result.confidence:.1%}

---

*Note: This is a research tool. All findings require validation by qualified pathologists.*
"""

            # Format evidence patches info
            evidence_text = "## Top Evidence Patches\n\n"
            for p in result.evidence_patches[:6]:
                evidence_text += f"**Patch {p['rank']}** (attention: {p['attention_weight']:.3f})\n"
                evidence_text += f"- Location: ({p['coordinates'][0]}, {p['coordinates'][1]})\n\n"

            # Create evidence gallery
            evidence_images = []
            for p in result.evidence_patches[:12]:
                if p.get("patch") is not None:
                    evidence_images.append(Image.fromarray(p["patch"]))

            return (
                Image.fromarray(overlay),
                pred_text,
                evidence_text,
                evidence_images[:12] if evidence_images else None,
                None  # Report placeholder
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None, f"Error: {str(e)}", None, None, None

    def generate_report_text():
        """Generate and display the MedGemma report."""
        if current_result["value"] is None:
            return "No analysis results available. Please analyze a slide first."

        result = current_result["value"]

        if result.report:
            # Generate formatted summary
            summary = atlas.reporter.generate_summary(result.report)
            return summary
        else:
            return "Report not generated. Enable 'Generate Report' during analysis."

    def export_results():
        """Export results to files."""
        if current_result["value"] is None:
            return None, "No results to export."

        result = current_result["value"]

        # Create temp directory for exports
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save report JSON
            report_path = tmpdir / "report.json"
            with open(report_path, "w") as f:
                json.dump(result.report, f, indent=2)

            # Save heatmap
            heatmap_path = tmpdir / "heatmap.png"
            result.save_heatmap(str(heatmap_path))

            return str(report_path), "Export successful!"

    # Build Gradio interface
    with gr.Blocks(
        title="Enso Atlas",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .evidence-gallery { max-height: 400px; overflow-y: auto; }
        """
    ) as app:

        gr.Markdown(
            """
            # Enso Atlas
            ### On-Prem Pathology Evidence Engine for Treatment-Response Insight

            Upload a whole-slide image (WSI) to analyze treatment response prediction with interpretable evidence.
            """,
            elem_classes=["main-header"]
        )

        with gr.Row():
            # Left column: Upload and controls
            with gr.Column(scale=1):
                slide_input = gr.File(
                    label="Upload WSI",
                    file_types=[".svs", ".ndpi", ".tiff", ".tif", ".mrxs"],
                )

                generate_report_checkbox = gr.Checkbox(
                    label="Generate MedGemma Report",
                    value=True,
                )

                analyze_btn = gr.Button(
                    "Analyze Slide",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("---")

                prediction_output = gr.Markdown(
                    label="Prediction",
                    value="Upload a slide to begin analysis.",
                )

            # Right column: Visualization
            with gr.Column(scale=2):
                heatmap_output = gr.Image(
                    label="Attention Heatmap",
                    type="pil",
                    interactive=False,
                )

        with gr.Row():
            # Evidence section
            with gr.Column():
                gr.Markdown("## Evidence Patches")
                evidence_text_output = gr.Markdown(
                    value="Evidence patches will appear after analysis.",
                )
                evidence_gallery = gr.Gallery(
                    label="Top Evidence Patches",
                    columns=6,
                    rows=2,
                    height=300,
                    elem_classes=["evidence-gallery"],
                )

        with gr.Row():
            # Report section
            with gr.Column():
                gr.Markdown("## Tumor Board Report")

                report_btn = gr.Button(
                    "Generate Report",
                    variant="secondary",
                )

                report_output = gr.Textbox(
                    label="MedGemma Report",
                    lines=20,
                    max_lines=30,
                )

        with gr.Row():
            # Export section
            with gr.Column():
                export_btn = gr.Button(
                    "📥 Export Results",
                    variant="secondary",
                )
                export_file = gr.File(label="Download")
                export_status = gr.Textbox(label="Status", lines=1)

        # Footer
        gr.Markdown(
            """
            ---
            **Safety Notice:** This is a research decision-support tool, not a medical device.
            All predictions and reports require validation by qualified pathologists.
            Do not use for standalone clinical decision-making.

            Built with [Path Foundation](https://developers.google.com/health-ai-developer-foundations/path-foundation)
            and [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
            from Google Health AI.
            """
        )

        # Wire up events
        analyze_btn.click(
            fn=analyze_slide,
            inputs=[slide_input, generate_report_checkbox],
            outputs=[
                heatmap_output,
                prediction_output,
                evidence_text_output,
                evidence_gallery,
                report_output,
            ],
        )

        report_btn.click(
            fn=generate_report_text,
            inputs=[],
            outputs=[report_output],
        )

        export_btn.click(
            fn=export_results,
            inputs=[],
            outputs=[export_file, export_status],
        )

    return app


def main():
    """Run the Gradio app."""
    import os

    config_path = os.environ.get("ENSO_CONFIG", "config/default.yaml")

    app = create_app(config_path)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
