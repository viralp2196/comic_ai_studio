#!/usr/bin/env python3
"""
Web interface for character dialog extraction
Upload scene images and get character dialog JSON
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import json
from pathlib import Path
import tempfile
from character_dialog_extractor import CharacterDialogExtractor

app = FastAPI(title="Character Dialog Extractor")

# Create directories
Path("output").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

# Mount static files
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.get("/", response_class=HTMLResponse)
async def upload_form():
    """Display upload form"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Character Dialog Extractor</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1000px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .container { 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .upload-area { 
                border: 2px dashed #007bff; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0; 
                border-radius: 10px;
                background-color: #f8f9fa;
            }
            .upload-area:hover { 
                background-color: #e9ecef; 
                border-color: #0056b3;
            }
            button { 
                background: #007bff; 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                font-size: 16px;
                transition: background-color 0.3s;
            }
            button:hover { background: #0056b3; }
            .feature-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature-item {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #007bff;
            }
            .feature-item h3 {
                margin-top: 0;
                color: #007bff;
            }
            input[type="file"] {
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100%;
                max-width: 400px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Character Dialog Extractor</h1>
            <p>Upload a comic scene image to automatically detect characters and extract their dialog</p>
            
            <div class="feature-list">
                <div class="feature-item">
                    <h3>🎯 Character Detection</h3>
                    <p>Automatically identifies character regions in comic scenes</p>
                </div>
                <div class="feature-item">
                    <h3>💬 Speech Bubble Recognition</h3>
                    <p>Detects and extracts text from speech bubbles</p>
                </div>
                <div class="feature-item">
                    <h3>🔗 Dialog Assignment</h3>
                    <p>Links dialog to the nearest character automatically</p>
                </div>
                <div class="feature-item">
                    <h3>📊 JSON Output</h3>
                    <p>Structured data format for easy integration</p>
                </div>
            </div>
            
            <form action="/extract-dialog" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <h3>📤 Upload Comic Scene</h3>
                    <input type="file" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit">🚀 Extract Character Dialog</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/extract-dialog", response_class=HTMLResponse)
async def extract_dialog(file: UploadFile = File(...)):
    """Extract character dialog from uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file
        upload_path = Path("uploads") / file.filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract character dialog
        extractor = CharacterDialogExtractor()
        scene_data = extractor.process_scene(str(upload_path))
        
        # Save results
        image_name = Path(file.filename).stem
        json_output = Path("output") / f"{image_name}_dialog.json"
        viz_output = Path("output") / f"{image_name}_visualization.jpg"
        
        extractor.save_results(scene_data, str(json_output))
        extractor.create_visualization(str(upload_path), scene_data, str(viz_output))
        
        # Generate HTML response
        html_content = generate_results_html(file.filename, scene_data, json_output, viz_output)
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .error {{ background: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                button {{ background: #6c757d; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }}
                button:hover {{ background: #545b62; }}
            </style>
        </head>
        <body>
            <h1>❌ Error Processing Image</h1>
            <div class="error">
                <strong>Error:</strong> {str(e)}
            </div>
            <a href="/"><button>← Back to Upload</button></a>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)


def generate_results_html(filename: str, scene_data: dict, json_path: Path, viz_path: Path) -> str:
    """Generate HTML for displaying extraction results"""
    
    # Create dialog summary
    dialog_summary = ""
    for character_id, dialogs in scene_data['dialog_by_character'].items():
        if dialogs:
            dialog_summary += f"""
            <div class="character-dialog">
                <h4>🎭 {character_id.replace('_', ' ').title()}</h4>
                <ul>
            """
            for dialog in dialogs:
                dialog_summary += f"<li>{dialog}</li>"
            dialog_summary += "</ul></div>"
    
    if not dialog_summary:
        dialog_summary = "<p>No dialog detected in this scene.</p>"
    
    # Create character summary
    character_summary = ""
    for i, character in enumerate(scene_data['characters'], 1):
        bbox = character['bbox']
        character_summary += f"""
        <div class="character-item">
            <strong>Character {i}</strong><br>
            Position: ({bbox['x']}, {bbox['y']})<br>
            Size: {bbox['width']}×{bbox['height']}<br>
            Area: {character['area']} pixels
        </div>
        """
    
    # Create speech bubble summary
    bubble_summary = ""
    for i, bubble in enumerate(scene_data['speech_bubbles'], 1):
        bbox = bubble['bbox']
        assigned = bubble.get('assigned_character', 'Unassigned')
        bubble_summary += f"""
        <div class="bubble-item">
            <strong>Bubble {i}</strong><br>
            Text: "{bubble['text']}"<br>
            Assigned to: {assigned}<br>
            Position: ({bbox['x']}, {bbox['y']})
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Character Dialog Results</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }}
            .container {{ 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .results {{ 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 8px; 
                border-left: 4px solid #28a745;
            }}
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }}
            .grid-item {{ 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                border: 1px solid #dee2e6;
            }}
            .character-dialog {{ 
                background: #e7f3ff; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px; 
                border-left: 4px solid #007bff;
            }}
            .character-item, .bubble-item {{ 
                background: #f8f9fa; 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px; 
                border: 1px solid #dee2e6;
            }}
            button {{ 
                background: #007bff; 
                color: white; 
                padding: 10px 20px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                margin: 5px; 
                text-decoration: none;
                display: inline-block;
            }}
            button:hover {{ background: #0056b3; }}
            .download-btn {{ background: #28a745; }}
            .download-btn:hover {{ background: #218838; }}
            .back-btn {{ background: #6c757d; }}
            .back-btn:hover {{ background: #545b62; }}
            .summary {{ 
                background: #d4edda; 
                padding: 20px; 
                border-radius: 8px; 
                margin: 20px 0; 
                border-left: 4px solid #28a745;
            }}
            .visualization {{ 
                text-align: center; 
                margin: 20px 0; 
            }}
            .visualization img {{ 
                max-width: 100%; 
                height: auto; 
                border: 1px solid #dee2e6; 
                border-radius: 8px;
            }}
            .json-preview {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                font-family: monospace;
                font-size: 12px;
                max-height: 300px;
                overflow-y: auto;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Character Dialog Extraction Results</h1>
            <a href="/"><button class="back-btn">← Upload Another Image</button></a>
            
            <div class="summary">
                <h3>📊 Extraction Summary</h3>
                <p><strong>Original Image:</strong> {filename}</p>
                <p><strong>Characters Detected:</strong> {len(scene_data['characters'])}</p>
                <p><strong>Speech Bubbles Found:</strong> {len(scene_data['speech_bubbles'])}</p>
                <p><strong>Dialog Extracted:</strong> {sum(len(dialogs) for dialogs in scene_data['dialog_by_character'].values())}</p>
            </div>
            
            <div class="results">
                <h3>💬 Extracted Dialog</h3>
                {dialog_summary}
            </div>
            
            <div class="visualization">
                <h3>🔍 Detection Visualization</h3>
                <p>Blue boxes: Characters | Red boxes: Speech bubbles | Green lines: Dialog assignments</p>
                <img src="/output/{viz_path.name}" alt="Detection Visualization">
            </div>
            
            <div class="grid">
                <div class="grid-item">
                    <h3>👥 Characters</h3>
                    {character_summary if character_summary else "<p>No characters detected</p>"}
                </div>
                
                <div class="grid-item">
                    <h3>💭 Speech Bubbles</h3>
                    {bubble_summary if bubble_summary else "<p>No speech bubbles detected</p>"}
                </div>
            </div>
            
            <div class="results">
                <h3>📁 Download Results</h3>
                <a href="/output/{json_path.name}" download><button class="download-btn">📄 Download JSON Data</button></a>
                <a href="/output/{viz_path.name}" download><button class="download-btn">🖼️ Download Visualization</button></a>
                <a href="/api/scene-data/{Path(filename).stem}"><button>🔗 View JSON API</button></a>
            </div>
            
            <div class="results">
                <h3>📋 JSON Preview</h3>
                <div class="json-preview">{json.dumps(scene_data, indent=2)}</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


@app.get("/api/scene-data/{image_name}")
async def get_scene_data(image_name: str):
    """API endpoint to get scene data as JSON"""
    json_path = Path("output") / f"{image_name}_dialog.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Scene data not found")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)
    
    return JSONResponse(content=scene_data)


if __name__ == "__main__":
    import uvicorn
    print("Starting Character Dialog Extractor Web Interface...")
    print("Open http://localhost:8003 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8003)