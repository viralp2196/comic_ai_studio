#!/usr/bin/env python3
"""
Test script to verify installation and dependencies
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import pytesseract
        print("✓ Pytesseract imported successfully")
    except ImportError as e:
        print(f"✗ Pytesseract import failed: {e}")
        print("  Install with: pip install pytesseract")
        return False
    
    try:
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    return True

def test_modules():
    """Test if our custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from panel_extractor import PanelExtractor
        print("✓ PanelExtractor imported successfully")
    except ImportError as e:
        print(f"✗ PanelExtractor import failed: {e}")
        return False
    
    try:
        from character_dialog_extractor import CharacterDialogExtractor
        print("✓ CharacterDialogExtractor imported successfully")
    except ImportError as e:
        print(f"✗ CharacterDialogExtractor import failed: {e}")
        return False
    
    return True

def test_ocr_initialization():
    """Test OCR engine availability"""
    print("\nTesting OCR engine...")
    
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image with text
        test_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        import cv2
        cv2.putText(test_img, "TEST", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(test_img)
        
        # Test OCR
        text = pytesseract.image_to_string(pil_img).strip()
        
        if "TEST" in text.upper():
            print("✓ OCR engine working correctly")
            return True
        else:
            print(f"⚠️  OCR engine detected but may have issues (got: '{text}')")
            return True
            
    except pytesseract.TesseractNotFoundError:
        print("✗ Tesseract OCR engine not found")
        print("  Install Tesseract:")
        print("    macOS: brew install tesseract")
        print("    Ubuntu: sudo apt-get install tesseract-ocr")
        print("    Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False

def create_sample_image():
    """Create a simple test image"""
    print("\nCreating sample test image...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Draw some rectangles to simulate panels/characters
        cv2.rectangle(img, (50, 50), (250, 180), (0, 0, 0), 2)
        cv2.rectangle(img, (300, 50), (550, 180), (0, 0, 0), 2)
        cv2.rectangle(img, (50, 220), (550, 350), (0, 0, 0), 2)
        
        # Add some text
        cv2.putText(img, "Test Comic Scene", (200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save test image
        test_path = Path("test_comic_scene.jpg")
        cv2.imwrite(str(test_path), img)
        print(f"✓ Sample image created: {test_path}")
        
        return str(test_path)
    
    except Exception as e:
        print(f"✗ Failed to create sample image: {e}")
        return None

def test_panel_extraction(image_path: str):
    """Test panel extraction functionality"""
    print(f"\nTesting panel extraction on {image_path}...")
    
    try:
        from panel_extractor import PanelExtractor
        
        extractor = PanelExtractor()
        panels = extractor.extract_panels(image_path, "test_output")
        
        print(f"✓ Panel extraction completed: {len(panels)} panels found")
        return True
    
    except Exception as e:
        print(f"✗ Panel extraction failed: {e}")
        return False

def test_dialog_extraction(image_path: str):
    """Test character dialog extraction functionality"""
    print(f"\nTesting character dialog extraction on {image_path}...")
    
    try:
        from character_dialog_extractor import CharacterDialogExtractor
        
        extractor = CharacterDialogExtractor()
        scene_data = extractor.process_scene(image_path)
        
        print(f"✓ Dialog extraction completed:")
        print(f"  Characters: {len(scene_data['characters'])}")
        print(f"  Speech bubbles: {len(scene_data['speech_bubbles'])}")
        
        return True
    
    except Exception as e:
        print(f"✗ Dialog extraction failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Comic Panel & Character Dialog Extractor - Installation Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Test custom modules
    if not test_modules():
        print("\n❌ Module tests failed. Check file structure.")
        sys.exit(1)
    
    # Test OCR initialization
    if not test_ocr_initialization():
        print("\n⚠️  OCR initialization failed, but other features may work.")
    
    # Create sample image
    sample_image = create_sample_image()
    
    if sample_image:
        # Test panel extraction
        if not test_panel_extraction(sample_image):
            print("\n⚠️  Panel extraction test failed.")
        
        # Test dialog extraction
        if not test_dialog_extraction(sample_image):
            print("\n⚠️  Dialog extraction test failed.")
    
    print("\n" + "=" * 60)
    print("✅ Installation test completed!")
    print("\nNext steps:")
    print("1. Try the panel extractor: python panel_extractor.py test_comic_scene.jpg")
    print("2. Try the dialog extractor: python character_dialog_extractor.py test_comic_scene.jpg")
    print("3. Start web interface: python dialog_web.py")
    print("4. Open http://localhost:8003 in your browser")
    print("\nTroubleshooting:")
    print("- If OCR fails, ensure Tesseract is installed and in PATH")
    print("- For better OCR results, use high-resolution images")
    print("- Check that speech bubbles have clear, readable text")

if __name__ == "__main__":
    main()