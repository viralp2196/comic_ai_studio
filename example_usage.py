#!/usr/bin/env python3
"""
Example usage of the Character Dialog Extractor
Demonstrates how to process comic scenes and work with the JSON output
"""

import json
from pathlib import Path
from character_dialog_extractor import CharacterDialogExtractor


def process_single_scene(image_path: str, output_dir: str = "output"):
    """Process a single scene and display results"""
    print(f"Processing scene: {image_path}")
    
    # Initialize extractor
    extractor = CharacterDialogExtractor()
    
    # Process the scene
    scene_data = extractor.process_scene(image_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate output filenames
    image_name = Path(image_path).stem
    json_output = output_path / f"{image_name}_dialog.json"
    viz_output = output_path / f"{image_name}_visualization.jpg"
    
    # Save results
    extractor.save_results(scene_data, str(json_output))
    extractor.create_visualization(image_path, scene_data, str(viz_output))
    
    # Display results
    print_scene_summary(scene_data)
    
    return scene_data


def process_multiple_scenes(image_directory: str, output_dir: str = "output"):
    """Process multiple scene images from a directory"""
    image_dir = Path(image_directory)
    
    if not image_dir.exists():
        print(f"Directory not found: {image_directory}")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {image_directory}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    all_scenes = []
    extractor = CharacterDialogExtractor()
    
    for image_file in image_files:
        print(f"\nProcessing: {image_file.name}")
        try:
            scene_data = extractor.process_scene(str(image_file))
            
            # Save individual results
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            json_output = output_path / f"{image_file.stem}_dialog.json"
            viz_output = output_path / f"{image_file.stem}_visualization.jpg"
            
            extractor.save_results(scene_data, str(json_output))
            extractor.create_visualization(str(image_file), scene_data, str(viz_output))
            
            all_scenes.append(scene_data)
            print_scene_summary(scene_data)
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    # Save combined results
    if all_scenes:
        combined_output = Path(output_dir) / "all_scenes_dialog.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump({
                'total_scenes': len(all_scenes),
                'scenes': all_scenes
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nCombined results saved to: {combined_output}")


def print_scene_summary(scene_data: dict):
    """Print a summary of the scene data"""
    print("-" * 50)
    print(f"Scene: {scene_data['scene_info']['image_name']}")
    print(f"Characters detected: {len(scene_data['characters'])}")
    print(f"Speech bubbles found: {len(scene_data['speech_bubbles'])}")
    
    # Print dialog by character
    if scene_data['dialog_by_character']:
        print("\nDialog extracted:")
        for character_id, dialogs in scene_data['dialog_by_character'].items():
            if dialogs:
                print(f"\n{character_id.replace('_', ' ').title()}:")
                for i, dialog in enumerate(dialogs, 1):
                    print(f"  {i}. \"{dialog}\"")
    else:
        print("No dialog detected")
    
    print("-" * 50)


def analyze_json_structure(json_file: str):
    """Analyze and display the structure of a dialog JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"JSON Structure Analysis: {json_file}")
    print("=" * 60)
    
    # Scene info
    scene_info = data['scene_info']
    print(f"Image: {scene_info['image_name']}")
    print(f"Dimensions: {scene_info['dimensions']['width']}x{scene_info['dimensions']['height']}")
    
    # Characters
    print(f"\nCharacters ({len(data['characters'])}):")
    for i, char in enumerate(data['characters'], 1):
        bbox = char['bbox']
        print(f"  {i}. {char['id']} - Position: ({bbox['x']}, {bbox['y']}) Size: {bbox['width']}x{bbox['height']}")
    
    # Speech bubbles
    print(f"\nSpeech Bubbles ({len(data['speech_bubbles'])}):")
    for i, bubble in enumerate(data['speech_bubbles'], 1):
        assigned = bubble.get('assigned_character', 'Unassigned')
        text_preview = bubble['text'][:50] + "..." if len(bubble['text']) > 50 else bubble['text']
        print(f"  {i}. Text: \"{text_preview}\" -> {assigned}")
    
    # Dialog summary
    print(f"\nDialog Summary:")
    total_dialog = 0
    for character_id, dialogs in data['dialog_by_character'].items():
        if dialogs:
            total_dialog += len(dialogs)
            print(f"  {character_id}: {len(dialogs)} dialog(s)")
    
    print(f"\nTotal dialog pieces: {total_dialog}")


def create_dialog_script(scene_data: dict) -> str:
    """Convert scene data to a readable script format"""
    script_lines = []
    script_lines.append(f"SCENE: {scene_data['scene_info']['image_name']}")
    script_lines.append("=" * 50)
    
    # Group dialog by character and create script
    for character_id, dialogs in scene_data['dialog_by_character'].items():
        if dialogs:
            character_name = character_id.replace('_', ' ').title()
            for dialog in dialogs:
                script_lines.append(f"{character_name}: {dialog}")
    
    if not any(dialogs for dialogs in scene_data['dialog_by_character'].values()):
        script_lines.append("[No dialog in this scene]")
    
    return "\n".join(script_lines)


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Character Dialog Extractor - Example Usage")
        print("=" * 50)
        print("Commands:")
        print("  python example_usage.py <image_file>           - Process single image")
        print("  python example_usage.py <directory>           - Process all images in directory")
        print("  python example_usage.py analyze <json_file>   - Analyze JSON structure")
        print("  python example_usage.py script <json_file>    - Convert to script format")
        print("\nExamples:")
        print("  python example_usage.py comic_scene.jpg")
        print("  python example_usage.py comic_pages/")
        print("  python example_usage.py analyze output/scene_dialog.json")
        print("  python example_usage.py script output/scene_dialog.json")
        return
    
    command = sys.argv[1]
    
    if command == "analyze" and len(sys.argv) > 2:
        # Analyze JSON structure
        json_file = sys.argv[2]
        if Path(json_file).exists():
            analyze_json_structure(json_file)
        else:
            print(f"JSON file not found: {json_file}")
    
    elif command == "script" and len(sys.argv) > 2:
        # Convert to script format
        json_file = sys.argv[2]
        if Path(json_file).exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            script = create_dialog_script(scene_data)
            print(script)
        else:
            print(f"JSON file not found: {json_file}")
    
    else:
        # Process image(s)
        path = Path(command)
        
        if path.is_file():
            # Single image
            process_single_scene(command)
        elif path.is_dir():
            # Directory of images
            process_multiple_scenes(command)
        else:
            print(f"Path not found: {command}")


if __name__ == "__main__":
    main()