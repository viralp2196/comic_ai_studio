#!/usr/bin/env python3
"""
Character Dialog Extractor
Detect characters and extract dialog from comic scenes
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import pytesseract
from PIL import Image, ImageDraw
import re


class CharacterDialogExtractor:
    """Extract characters and dialog from comic scenes"""
    
    def __init__(self, 
                 min_speech_bubble_area: int = 500,
                 max_speech_bubble_area: int = 50000,
                 text_confidence_threshold: int = 60,
                 min_character_area: int = 1000,
                 character_aspect_ratio_range: tuple = (0.3, 3.0)):
        """
        Initialize the extractor with configurable parameters
        
        Args:
            min_speech_bubble_area: Minimum area for speech bubble detection
            max_speech_bubble_area: Maximum area for speech bubble detection
            text_confidence_threshold: Tesseract confidence threshold (0-100)
            min_character_area: Minimum area for character detection
            character_aspect_ratio_range: Valid aspect ratio range for characters
        """
        # Dialog detection parameters
        self.min_speech_bubble_area = min_speech_bubble_area
        self.max_speech_bubble_area = max_speech_bubble_area
        self.text_confidence_threshold = text_confidence_threshold
        
        # Character detection parameters
        self.min_character_area = min_character_area
        self.character_aspect_ratio_range = character_aspect_ratio_range
        
        # OCR configuration for comic text
        # PSM 6: Uniform block of text
        # OEM 3: Default OCR Engine Mode
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?\'\" -'
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def detect_speech_bubbles(self, img: np.ndarray) -> List[Dict]:
        """Detect speech bubbles in the image using multiple methods"""
        speech_bubbles = []
        
        # Method 1: Detect white/light colored speech bubbles
        bubbles_1 = self._detect_light_speech_bubbles(img)
        speech_bubbles.extend(bubbles_1)
        
        # Method 2: Detect dark outlined speech bubbles
        bubbles_2 = self._detect_outlined_speech_bubbles(img)
        speech_bubbles.extend(bubbles_2)
        
        # Method 3: Detect classic comic speech bubbles (white with black border)
        bubbles_3 = self._detect_classic_speech_bubbles(img)
        speech_bubbles.extend(bubbles_3)
        
        # Remove duplicates based on overlap
        speech_bubbles = self._remove_duplicate_bubbles(speech_bubbles)
        
        return speech_bubbles
    
    def _detect_light_speech_bubbles(self, img: np.ndarray) -> List[Dict]:
        """Detect light/white speech bubbles"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use multiple thresholds to catch different brightness levels
        thresholds = [180, 200, 220]
        all_bubbles = []
        
        for thresh_val in thresholds:
            # Threshold to find white/light areas
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                if self.min_speech_bubble_area <= area <= self.max_speech_bubble_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    extent = area / (w * h)
                    
                    # More lenient criteria for speech bubbles
                    if 0.2 <= aspect_ratio <= 6.0 and extent > 0.3:
                        # Check if it's likely a speech bubble by analyzing the region
                        region = gray[y:y+h, x:x+w]
                        if self._is_likely_speech_bubble(region):
                            all_bubbles.append({
                                'id': f'light_bubble_{thresh_val}_{i}',
                                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'center': {'x': x + w//2, 'y': y + h//2},
                                'type': 'light'
                            })
        
        return all_bubbles
    
    def _detect_classic_speech_bubbles(self, img: np.ndarray) -> List[Dict]:
        """Detect classic comic speech bubbles - white interior with dark border"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find very bright regions (white speech bubble interiors)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Find dark regions (borders and text)
        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Combine to find regions that have both white areas and dark borders/text
        # Dilate the dark mask to connect nearby dark pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_dilated = cv2.dilate(dark_mask, kernel, iterations=1)
        
        # Find regions that are mostly white but have some dark content
        combined = cv2.bitwise_and(white_mask, cv2.bitwise_not(dark_dilated))
        
        # Use morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if self.min_speech_bubble_area <= area <= self.max_speech_bubble_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it's likely a speech bubble
                if 0.5 <= aspect_ratio <= 5.0:
                    # Additional validation: check if region has good contrast
                    region = gray[y:y+h, x:x+w]
                    if region.size > 0:
                        brightness_std = np.std(region)
                        mean_brightness = np.mean(region)
                        
                        # Speech bubbles should be bright with some contrast (text)
                        if mean_brightness > 200 and brightness_std > 15:
                            bubbles.append({
                                'id': f'classic_bubble_{i}',
                                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'center': {'x': x + w//2, 'y': y + h//2},
                                'type': 'classic'
                            })
        
        return bubbles
    
    def _detect_outlined_speech_bubbles(self, img: np.ndarray) -> List[Dict]:
        """Detect speech bubbles with dark outlines"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find outlines
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if self.min_speech_bubble_area <= area <= self.max_speech_bubble_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it's likely a speech bubble
                if 0.2 <= aspect_ratio <= 5.0:
                    # Check if the interior is lighter than the outline
                    if self._has_light_interior(gray, x, y, w, h):
                        bubbles.append({
                            'id': f'outlined_bubble_{i}',
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': {'x': x + w//2, 'y': y + h//2},
                            'type': 'outlined'
                        })
        
        return bubbles
    
    def _is_likely_speech_bubble(self, region: np.ndarray) -> bool:
        """Check if a region is likely a speech bubble"""
        if region.size == 0:
            return False
        
        # Check brightness - speech bubbles are usually light
        avg_brightness = np.mean(region)
        if avg_brightness < 150:
            return False
        
        # Look for dark pixels that could be text or borders
        dark_pixels = np.sum(region < 100)
        medium_pixels = np.sum((region >= 100) & (region < 200))
        light_pixels = np.sum(region >= 200)
        total_pixels = region.size
        
        dark_ratio = dark_pixels / total_pixels
        light_ratio = light_pixels / total_pixels
        
        # Speech bubbles should have:
        # - Mostly light pixels (background)
        # - Some dark pixels (text/border)
        # - Good contrast
        return (light_ratio > 0.4 and 
                dark_ratio > 0.05 and 
                dark_ratio < 0.4 and
                (np.max(region) - np.min(region)) > 50)  # Good contrast
    
    def _contains_text_region(self, region: np.ndarray) -> bool:
        """Check if a region likely contains text"""
        if region.size == 0:
            return False
        
        # Look for dark pixels that could be text
        dark_pixels = np.sum(region < 100)
        total_pixels = region.size
        
        # If 5-50% of pixels are dark, it might contain text
        dark_ratio = dark_pixels / total_pixels
        return 0.05 <= dark_ratio <= 0.5
    
    def _has_light_interior(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Check if the interior of a region is lighter than its border"""
        if w < 20 or h < 20:
            return False
        
        # Sample interior region (avoid edges)
        margin = min(w//4, h//4, 10)
        interior = gray[y+margin:y+h-margin, x+margin:x+w-margin]
        
        if interior.size == 0:
            return False
        
        # Sample border region
        border_top = gray[y:y+margin, x:x+w]
        border_bottom = gray[y+h-margin:y+h, x:x+w]
        border_left = gray[y:y+h, x:x+margin]
        border_right = gray[y:y+h, x+w-margin:x+w]
        
        border_pixels = np.concatenate([
            border_top.flatten(), border_bottom.flatten(),
            border_left.flatten(), border_right.flatten()
        ])
        
        if border_pixels.size == 0:
            return False
        
        # Compare average brightness
        interior_avg = np.mean(interior)
        border_avg = np.mean(border_pixels)
        
        return interior_avg > border_avg + 20  # Interior should be significantly lighter
    
    def _remove_duplicate_bubbles(self, bubbles: List[Dict]) -> List[Dict]:
        """Remove overlapping speech bubbles"""
        if len(bubbles) <= 1:
            return bubbles
        
        # Sort by area (keep larger bubbles)
        bubbles.sort(key=lambda b: b['area'], reverse=True)
        
        filtered_bubbles = []
        for bubble in bubbles:
            is_duplicate = False
            
            for existing in filtered_bubbles:
                if self._bubbles_overlap(bubble, existing, threshold=0.5):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_bubbles.append(bubble)
        
        return filtered_bubbles
    
    def _bubbles_overlap(self, bubble1: Dict, bubble2: Dict, threshold: float = 0.5) -> bool:
        """Check if two bubbles overlap significantly"""
        bbox1 = bubble1['bbox']
        bbox2 = bubble2['bbox']
        
        # Calculate intersection
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
        y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection_area = (x2 - x1) * (y2 - y1)
        smaller_area = min(bubble1['area'], bubble2['area'])
        
        return intersection_area / smaller_area > threshold
    
    def extract_text_from_bubble(self, img: np.ndarray, bubble: Dict) -> str:
        """Extract text from a speech bubble using OCR with multiple preprocessing methods"""
        bbox = bubble['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Extract bubble region with padding
        padding = 15
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        bubble_img = img[y1:y2, x1:x2]
        
        if bubble_img.size == 0:
            return ""
        
        # Try multiple preprocessing approaches
        text_results = []
        
        # Method 1: Standard preprocessing
        text1 = self._extract_with_standard_preprocessing(bubble_img)
        if text1:
            text_results.append(text1)
        
        # Method 2: Inverted preprocessing (for dark text on light background)
        text2 = self._extract_with_inverted_preprocessing(bubble_img)
        if text2:
            text_results.append(text2)
        
        # Method 3: High contrast preprocessing
        text3 = self._extract_with_high_contrast_preprocessing(bubble_img)
        if text3:
            text_results.append(text3)
        
        # Return the longest/best result
        if text_results:
            # Choose the result with the most words
            best_text = max(text_results, key=lambda t: len(t.split()))
            return best_text
        
        return ""
    
    def _extract_with_standard_preprocessing(self, bubble_img: np.ndarray) -> str:
        """Standard OCR preprocessing"""
        try:
            gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return self._run_ocr(thresh)
        except:
            return ""
    
    def _extract_with_inverted_preprocessing(self, bubble_img: np.ndarray) -> str:
        """OCR preprocessing for dark text on light background"""
        try:
            gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
            
            # Invert if background is light
            if np.mean(gray) > 127:
                gray = 255 - gray
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return self._run_ocr(thresh)
        except:
            return ""
    
    def _extract_with_high_contrast_preprocessing(self, bubble_img: np.ndarray) -> str:
        """High contrast OCR preprocessing"""
        try:
            gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
            
            # Apply strong contrast enhancement
            alpha = 2.0  # Contrast control
            beta = -50   # Brightness control
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
            # Bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return self._run_ocr(thresh)
        except:
            return ""
    
    def _run_ocr(self, processed_img: np.ndarray) -> str:
        """Run OCR on preprocessed image"""
        try:
            # Scale up for better OCR
            scale_factor = 3
            height, width = processed_img.shape
            scaled = cv2.resize(
                processed_img, 
                (width * scale_factor, height * scale_factor), 
                interpolation=cv2.INTER_CUBIC
            )
            
            # Try different PSM modes for comic text
            configs = [
                '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\'" -',  # Uniform block
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\'" -',  # Single line
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\'" -',  # Single word
                '--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\'" -', # Raw line
                '--oem 3 --psm 6',  # Without whitelist
                '--oem 3 --psm 7',  # Without whitelist
            ]
            
            best_text = ""
            best_score = 0
            
            for config in configs:
                try:
                    # Simple text extraction
                    simple_text = pytesseract.image_to_string(scaled, config=config).strip()
                    if simple_text:
                        # Clean up text
                        cleaned_text = re.sub(r'[^\w\s\.\!\?\,\-\'\"]', '', simple_text)
                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                        
                        # Score based on length and character variety
                        score = len(cleaned_text) + len(set(cleaned_text.replace(' ', '')))
                        
                        if score > best_score and len(cleaned_text) > 0:
                            best_text = cleaned_text
                            best_score = score
                    
                    # Also try with confidence data
                    data = pytesseract.image_to_data(
                        scaled, config=config, output_type=pytesseract.Output.DICT
                    )
                    
                    # Combine text with lower confidence threshold
                    text_parts = []
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > 10:  # Very low threshold for comic text
                            text = data['text'][i].strip()
                            if text:
                                text_parts.append(text)
                    
                    if text_parts:
                        combined_text = ' '.join(text_parts)
                        cleaned_text = re.sub(r'[^\w\s\.\!\?\,\-\'\"]', '', combined_text)
                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                        
                        score = len(cleaned_text) + len(set(cleaned_text.replace(' ', '')))
                        
                        if score > best_score and len(cleaned_text) > 0:
                            best_text = cleaned_text
                            best_score = score
                
                except:
                    continue
            
            return best_text
            
        except Exception as e:
            return ""
    
    def detect_characters(self, img: np.ndarray) -> List[Dict]:
        """Detect character regions in the image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use color information to better distinguish characters
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin tones and colored regions (characters)
        # Skin tone range in HSV
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Also look for other colored regions (clothing, hair)
        # Exclude very light colors (likely speech bubbles)
        _, light_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        colored_mask = cv2.bitwise_not(light_mask)
        
        # Combine masks
        character_mask = cv2.bitwise_or(skin_mask, colored_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        character_mask = cv2.morphologyEx(character_mask, cv2.MORPH_CLOSE, kernel)
        character_mask = cv2.morphologyEx(character_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(character_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area > self.min_character_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Filter character candidates by aspect ratio and other criteria
                if self.character_aspect_ratio_range[0] <= aspect_ratio <= self.character_aspect_ratio_range[1]:
                    # Additional check: avoid regions that are too light (likely speech bubbles)
                    region = gray[y:y+h, x:x+w]
                    avg_brightness = np.mean(region)
                    
                    if avg_brightness < 180:  # Characters should have some darker areas
                        characters.append({
                            'id': f'character_{len(characters)}',
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': {'x': x + w//2, 'y': y + h//2}
                        })
        
        return characters
    
    def assign_dialog_to_characters(self, characters: List[Dict], speech_bubbles: List[Dict]) -> List[Dict]:
        """Assign speech bubbles to nearest characters"""
        for bubble in speech_bubbles:
            bubble['assigned_character'] = None
            bubble['distance_to_character'] = float('inf')
            
            bubble_center = bubble['center']
            
            for character in characters:
                char_center = character['center']
                
                # Calculate distance between bubble and character
                distance = np.sqrt(
                    (bubble_center['x'] - char_center['x'])**2 + 
                    (bubble_center['y'] - char_center['y'])**2
                )
                
                # Assign to closest character
                if distance < bubble['distance_to_character']:
                    bubble['distance_to_character'] = distance
                    bubble['assigned_character'] = character['id']
        
        return speech_bubbles
    
    def process_scene(self, image_path: str) -> Dict[str, Any]:
        """Process a scene image and extract character dialog data"""
        img = self.load_image(image_path)
        
        # Detect speech bubbles
        speech_bubbles = self.detect_speech_bubbles(img)
        
        # Extract text from each speech bubble
        for bubble in speech_bubbles:
            bubble['text'] = self.extract_text_from_bubble(img, bubble)
        
        # Detect characters
        characters = self.detect_characters(img)
        
        # Assign dialog to characters
        speech_bubbles = self.assign_dialog_to_characters(characters, speech_bubbles)
        
        # Create scene data structure
        scene_data = {
            'scene_info': {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'dimensions': {
                    'width': img.shape[1],
                    'height': img.shape[0]
                }
            },
            'characters': characters,
            'speech_bubbles': speech_bubbles,
            'dialog_by_character': self._group_dialog_by_character(characters, speech_bubbles)
        }
        
        return scene_data
    
    def _group_dialog_by_character(self, characters: List[Dict], speech_bubbles: List[Dict]) -> Dict[str, List[str]]:
        """Group dialog text by character"""
        dialog_by_character = {}
        
        # Initialize with empty lists for all characters
        for character in characters:
            dialog_by_character[character['id']] = []
        
        # Add unassigned dialog category
        dialog_by_character['unassigned'] = []
        
        # Group speech bubbles by assigned character
        for bubble in speech_bubbles:
            if bubble['text']:  # Only include bubbles with text
                if bubble['assigned_character']:
                    dialog_by_character[bubble['assigned_character']].append(bubble['text'])
                else:
                    dialog_by_character['unassigned'].append(bubble['text'])
        
        return dialog_by_character
    
    def save_results(self, scene_data: Dict[str, Any], output_path: str):
        """Save extraction results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
    
    def create_visualization(self, image_path: str, scene_data: Dict[str, Any], output_path: str):
        """Create a visualization showing detected characters and speech bubbles"""
        img = cv2.imread(image_path)
        
        # Draw character bounding boxes in blue
        for character in scene_data['characters']:
            bbox = character['bbox']
            cv2.rectangle(img, 
                         (bbox['x'], bbox['y']), 
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                         (255, 0, 0), 2)
            cv2.putText(img, character['id'], 
                       (bbox['x'], bbox['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw speech bubble bounding boxes in red
        for bubble in scene_data['speech_bubbles']:
            bbox = bubble['bbox']
            cv2.rectangle(img, 
                         (bbox['x'], bbox['y']), 
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                         (0, 0, 255), 2)
            
            # Show assignment line if character is assigned
            if bubble['assigned_character']:
                # Find the assigned character
                assigned_char = next(
                    (char for char in scene_data['characters'] 
                     if char['id'] == bubble['assigned_character']), None
                )
                if assigned_char:
                    bubble_center = (bubble['center']['x'], bubble['center']['y'])
                    char_center = (assigned_char['center']['x'], assigned_char['center']['y'])
                    cv2.line(img, bubble_center, char_center, (0, 255, 0), 1)
        
        # Save visualization
        cv2.imwrite(output_path, img)


def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python character_dialog_extractor.py <image_path> [output_dir]")
        print("Example: python character_dialog_extractor.py comic_scene.jpg output")
        return
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Initialize extractor
        extractor = CharacterDialogExtractor()
        
        print(f"Processing scene: {image_path}")
        print(f"Output directory: {output_dir}")
        print("-" * 50)
        
        # Process the scene
        scene_data = extractor.process_scene(image_path)
        
        # Generate output filenames
        image_name = Path(image_path).stem
        json_output = output_path / f"{image_name}_dialog.json"
        viz_output = output_path / f"{image_name}_visualization.jpg"
        
        # Save results
        extractor.save_results(scene_data, str(json_output))
        extractor.create_visualization(image_path, scene_data, str(viz_output))
        
        # Print summary
        print(f"Characters detected: {len(scene_data['characters'])}")
        print(f"Speech bubbles detected: {len(scene_data['speech_bubbles'])}")
        print(f"Dialog extracted: {sum(len(dialogs) for dialogs in scene_data['dialog_by_character'].values())}")
        print("-" * 50)
        print(f"Results saved to: {json_output}")
        print(f"Visualization saved to: {viz_output}")
        
        # Print dialog summary
        if scene_data['dialog_by_character']:
            print("\nDialog Summary:")
            for character_id, dialogs in scene_data['dialog_by_character'].items():
                if dialogs:
                    print(f"\n{character_id}:")
                    for i, dialog in enumerate(dialogs, 1):
                        print(f"  {i}. {dialog}")
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()