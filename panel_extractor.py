#!/usr/bin/env python3
"""
Comic Panel Extractor
Extract individual panels from comic images
"""

import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path


class PanelExtractor:
    """Extract and save comic panels from images"""
    
    def __init__(self, min_panel_size_ratio: float = 0.05):
        self.min_panel_size_ratio = min_panel_size_ratio
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for panel detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh
    
    def detect_panels(self, img: np.ndarray) -> list:
        """Detect panels in the image"""
        height, width = img.shape[:2]
        min_width = int(width * self.min_panel_size_ratio)
        min_height = int(height * self.min_panel_size_ratio)
        
        thresh = self.preprocess_image(img)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        panels = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_width and h >= min_height:
                panels.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })
        
        # Sort panels by position (top to bottom, left to right)
        panels = sorted(panels, key=lambda p: (p['y'] // 100, p['x']))
        return panels
    
    def extract_panels(self, image_path: str, output_dir: str = "output"):
        """Extract panels from image and save them as separate files"""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load image
        img = self.load_image(image_path)
        
        # Detect panels
        panels = self.detect_panels(img)
        
        if not panels:
            print("No panels detected in the image")
            return []
        
        # Extract and save each panel
        saved_panels = []
        input_filename = Path(image_path).stem
        
        for i, panel in enumerate(panels, 1):
            x, y, w, h = panel['x'], panel['y'], panel['width'], panel['height']
            
            # Extract panel from image
            panel_img = img[y:y+h, x:x+w]
            
            # Convert BGR to RGB for PIL
            panel_rgb = cv2.cvtColor(panel_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(panel_rgb)
            
            # Save panel
            panel_filename = f"{input_filename}_panel_{i:02d}.png"
            panel_path = output_path / panel_filename
            pil_img.save(panel_path)
            
            saved_panels.append({
                'panel_number': i,
                'filename': panel_filename,
                'path': str(panel_path),
                'dimensions': f"{w}x{h}",
                'position': f"({x}, {y})"
            })
            
            print(f"Saved panel {i}: {panel_filename} ({w}x{h})")
        
        return saved_panels


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python panel_extractor.py <image_path> [output_dir]")
        print("Example: python panel_extractor.py comic.jpg output")
        return
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    try:
        extractor = PanelExtractor()
        
        print(f"Processing image: {image_path}")
        print(f"Output directory: {output_dir}")
        print("-" * 50)
        
        panels = extractor.extract_panels(image_path, output_dir)
        
        print("-" * 50)
        print(f"Extraction complete!")
        print(f"Total panels extracted: {len(panels)}")
        print(f"Panels saved to: {Path(output_dir).absolute()}")
        
        if panels:
            print("\nPanel details:")
            for panel in panels:
                print(f"  {panel['filename']} - {panel['dimensions']} at {panel['position']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()