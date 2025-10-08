# Comic Panel Extractor

Extract individual panels from comic images using computer vision.

## Features

- **Panel Detection**: Automatically detect comic panels using OpenCV
- **Image Extraction**: Save each panel as a separate image file
- **Command Line Interface**: Simple CLI for batch processing
- **Web Interface**: Upload images and download extracted panels

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy pillow fastapi uvicorn python-multipart
```

## Usage

### Command Line

```bash
# Extract panels from an image
python panel_extractor.py comic.jpg

# Specify output directory
python panel_extractor.py comic.jpg panels_output
```

### Web Interface

```bash
# Start web server
python panel_web.py

# Open http://localhost:8002 in your browser
```

### Example Output

```
Processing image: comic.jpg
Output directory: output
--------------------------------------------------
Saved panel 1: comic_panel_01.png (300x200)
Saved panel 2: comic_panel_02.png (250x180)
Saved panel 3: comic_panel_03.png (280x220)
--------------------------------------------------
Extraction complete!
Total panels extracted: 3
```

## How It Works

1. **Image Preprocessing**: Convert to grayscale and apply adaptive thresholding
2. **Contour Detection**: Find panel boundaries using OpenCV contours
3. **Panel Filtering**: Remove panels that are too small
4. **Panel Extraction**: Crop and save each panel as a separate image

## Configuration

Adjust panel detection sensitivity by modifying `min_panel_size_ratio` in the PanelExtractor class:

```python
extractor = PanelExtractor(min_panel_size_ratio=0.05)  # Detect smaller panels
```

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Pillow
- FastAPI (for web interface)

## License

MIT License