# Comic Panel & Character Dialog Extractor

Extract individual panels from comic images and detect characters with their dialog using computer vision and OCR.

## Features

### Panel Extraction
- **Panel Detection**: Automatically detect comic panels using OpenCV
- **Image Extraction**: Save each panel as a separate image file
- **Command Line Interface**: Simple CLI for batch processing
- **Web Interface**: Upload images and download extracted panels

### Character Dialog Extraction
- **Character Detection**: Automatically identify character regions in comic scenes
- **Speech Bubble Recognition**: Detect and extract text from speech bubbles using OCR
- **Dialog Assignment**: Link dialog to the nearest character automatically
- **JSON Output**: Structured data format with character and dialog information
- **Visualization**: Generate annotated images showing detection results

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# OR install manually:
# pip install opencv-python numpy pillow fastapi uvicorn python-multipart pytesseract

# Install Tesseract OCR engine (required for pytesseract)
# On macOS:
brew install tesseract

# On Ubuntu/Debian:
# sudo apt-get install tesseract-ocr

# On Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Panel Extraction

#### Command Line
```bash
# Extract panels from an image
python panel_extractor.py comic.jpg

# Specify output directory
python panel_extractor.py comic.jpg panels_output
```

#### Web Interface
```bash
# Start panel extraction web server
python panel_web.py

# Open http://localhost:8002 in your browser
```

### Character Dialog Extraction

#### Command Line
```bash
# Extract character dialog from a scene
python character_dialog_extractor.py comic_scene.jpg

# Specify output directory
python character_dialog_extractor.py comic_scene.jpg output

# Process multiple scenes and analyze results
python example_usage.py comic_scenes/
python example_usage.py analyze output/scene_dialog.json
python example_usage.py script output/scene_dialog.json
```

#### Web Interface
```bash
# Start character dialog extraction web server
python dialog_web.py

# Open http://localhost:8003 in your browser
```

### Example Output

#### Panel Extraction
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

#### Character Dialog Extraction
```
Processing scene: comic_scene.jpg
Output directory: output
--------------------------------------------------
Characters detected: 2
Speech bubbles detected: 3
Dialog extracted: 3
--------------------------------------------------
Results saved to: output/comic_scene_dialog.json
Visualization saved to: output/comic_scene_visualization.jpg

Dialog Summary:

Character 0:
  1. "Hello there!"
  2. "How are you doing?"

Character 1:
  1. "I'm doing great, thanks!"
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

## JSON Output Structure

The character dialog extractor generates structured JSON data with the following format:

```json
{
  "scene_info": {
    "image_path": "comic_scene.jpg",
    "image_name": "comic_scene.jpg",
    "dimensions": {
      "width": 800,
      "height": 600
    }
  },
  "characters": [
    {
      "id": "character_0",
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 300
      },
      "area": 60000,
      "aspect_ratio": 0.67,
      "center": {
        "x": 200,
        "y": 300
      }
    }
  ],
  "speech_bubbles": [
    {
      "id": "bubble_0",
      "bbox": {
        "x": 50,
        "y": 50,
        "width": 150,
        "height": 80
      },
      "area": 12000,
      "aspect_ratio": 1.875,
      "center": {
        "x": 125,
        "y": 90
      },
      "text": "Hello there!",
      "assigned_character": "character_0",
      "distance_to_character": 45.2
    }
  ],
  "dialog_by_character": {
    "character_0": [
      "Hello there!",
      "How are you doing?"
    ],
    "character_1": [
      "I'm doing great, thanks!"
    ],
    "unassigned": []
  }
}
```

## API Endpoints

When using the web interface, you can access the following API endpoints:

- `GET /api/scene-data/{image_name}` - Get scene data as JSON
- `POST /extract-dialog` - Upload image and extract dialog
- `GET /output/{filename}` - Download generated files

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Pillow
- FastAPI (for web interface)
- Pytesseract (for text extraction)
- Tesseract OCR engine

## License

MIT License