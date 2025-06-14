import subprocess
from pathlib import Path

def convert_visio_to_pdf(visio_path, output_dir=None):
    """
    Converts a Visio file (.vsd, .vsdx) to PDF using LibreOffice.
    Returns the path to the generated PDF.
    """
    visio_path = Path(visio_path)
    output_dir = Path(output_dir) if output_dir else visio_path.parent
    result = subprocess.run([
        "soffice", "--headless", "--convert-to", "pdf", str(visio_path), "--outdir", str(output_dir)
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice conversion failed: {result.stderr.decode()}")
    pdf_path = output_dir / (visio_path.stem + ".pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not created: {pdf_path}")
    return pdf_path 