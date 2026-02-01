"""
CLI for Enso Atlas.

Commands:
- analyze: Analyze a single slide
- batch: Batch analyze slides in a directory
- train: Train the MIL head
- serve: Start the Gradio UI
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="enso-atlas",
    help="On-Prem Pathology Evidence Engine for Treatment-Response Insight"
)
console = Console()


@app.command()
def analyze(
    slide: Path = typer.Argument(..., help="Path to WSI file"),
    output: Path = typer.Option("outputs/", help="Output directory"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
    no_report: bool = typer.Option(False, help="Skip MedGemma report generation"),
):
    """Analyze a single whole-slide image."""
    from .core import EnsoAtlas

    console.print(f"[bold blue]Analyzing slide:[/] {slide.name}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        atlas = EnsoAtlas.from_config(str(config))

        progress.update(task, description="Processing slide...")
        result = atlas.analyze(str(slide), generate_report=not no_report)

        # Save outputs
        output.mkdir(parents=True, exist_ok=True)
        slide_name = slide.stem

        progress.update(task, description="Saving results...")
        result.save_heatmap(output / f"{slide_name}_heatmap.png")
        result.save_report(output / f"{slide_name}_report.json")

    console.print(f"\n[bold green]Analysis complete![/]")
    console.print(f"  Prediction: [bold]{result.label}[/] (score: {result.score:.3f})")
    console.print(f"  Confidence: {result.confidence:.1%}")
    console.print(f"\n  Results saved to: {output}/")


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing WSI files"),
    output_dir: Path = typer.Option("outputs/", help="Output directory"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
    pattern: str = typer.Option("*.svs", help="Glob pattern for slide files"),
):
    """Batch analyze slides in a directory."""
    from .core import EnsoAtlas

    console.print(f"[bold blue]Batch analyzing slides in:[/] {input_dir}")

    atlas = EnsoAtlas.from_config(str(config))
    results = atlas.batch_analyze(str(input_dir), str(output_dir), pattern)

    console.print(f"\n[bold green]Batch analysis complete![/]")
    console.print(f"  Processed {len(results)} slides")
    console.print(f"  Results saved to: {output_dir}/")


@app.command()
def train(
    data_dir: Path = typer.Argument(..., help="Directory with embeddings and labels"),
    output: Path = typer.Option("models/clam.pt", help="Output model path"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
    val_split: float = typer.Option(0.2, help="Validation split ratio"),
):
    """Train the CLAM MIL head."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from .config import AtlasConfig
    from .mil.clam import CLAMClassifier

    console.print(f"[bold blue]Training MIL head[/]")

    # Load config
    config = AtlasConfig.from_yaml(str(config))

    # Load data
    # Expects: data_dir/embeddings/*.npy and data_dir/labels.csv
    embeddings_dir = data_dir / "embeddings"
    labels_file = data_dir / "labels.csv"

    if not embeddings_dir.exists():
        console.print("[red]Error: embeddings directory not found[/]")
        raise typer.Exit(1)

    if not labels_file.exists():
        console.print("[red]Error: labels.csv not found[/]")
        raise typer.Exit(1)

    # Load labels
    import pandas as pd
    labels_df = pd.read_csv(labels_file)

    # Load embeddings
    embeddings_list = []
    labels = []

    for _, row in labels_df.iterrows():
        emb_path = embeddings_dir / f"{row['slide_id']}.npy"
        if emb_path.exists():
            embeddings_list.append(np.load(emb_path))
            labels.append(row['label'])

    console.print(f"Loaded {len(embeddings_list)} slides")

    # Split data
    train_emb, val_emb, train_labels, val_labels = train_test_split(
        embeddings_list, labels, test_size=val_split, stratify=labels, random_state=42
    )

    # Train
    classifier = CLAMClassifier(config.mil)
    history = classifier.fit(train_emb, train_labels, val_emb, val_labels)

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(output)

    console.print(f"\n[bold green]Training complete![/]")
    console.print(f"  Final val AUC: {history['val_auc'][-1]:.3f}")
    console.print(f"  Model saved to: {output}")


@app.command()
def serve(
    config: Path = typer.Option("config/default.yaml", help="Config file"),
    port: int = typer.Option(7860, help="Server port"),
    share: bool = typer.Option(False, help="Create public URL"),
):
    """Start the Gradio web interface."""
    from .ui.app import create_app

    console.print(f"[bold blue]Starting Enso Atlas UI[/]")
    console.print(f"  Port: {port}")
    console.print(f"  Config: {config}")

    app = create_app(str(config))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
    )


@app.command()
def info():
    """Show system information and status."""
    import torch

    console.print("[bold blue]Enso Atlas System Info[/]\n")

    # Python info
    import sys
    console.print(f"Python: {sys.version}")

    # PyTorch info
    console.print(f"PyTorch: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        console.print(f"CUDA: {torch.version.cuda}")
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        console.print("CUDA: Not available")

    # MPS info (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print("MPS (Apple Silicon): Available")

    # OpenSlide info
    try:
        import openslide
        console.print(f"OpenSlide: {openslide.__version__}")
    except ImportError:
        console.print("OpenSlide: Not installed")

    console.print("\n[bold green]Ready for analysis![/]")


if __name__ == "__main__":
    app()
