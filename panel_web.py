#!/usr/bin/env python3
"""
Simple web interface for panel extraction
Upload image and download extracted panels
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import zipfile
from pathlib import Path
import tempfile
from panel_extractor import PanelExtractor

app = FastAPI(title="Panel Extractor")

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
        <title>Comic Panel Extractor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .results { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .panel-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .panel-item { border: 1px solid #ddd; padding: 10px; background: white; text-align: center; }
            .panel-item img { max-width: 100%; height: auto; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #0056b3; }
            .download-btn { background: #28a745; }
            .download-btn:hover { background: #218838; }
        </style>
    </head>
    <body>
        <h1>Comic Panel Extractor</h1>
        <p>Upload a comic image to extract individual panels</p>
        
        <form action="/extract-panels" method="post" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" name="file" accept="image/*" required>
                <br><br>
                <button type="submit">Extract Panels</button>
            </div>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/extract-panels", response_class=HTMLResponse)
async def extract_panels(file: UploadFile = File(...)):
    """Extract panels from uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file
        upload_path = Path("uploads") / file.filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract panels
        extractor = PanelExtractor()
        panels = extractor.extract_panels(str(upload_path), "output")
        
        # Generate HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Panel Extraction Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .results {{ background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .panel-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .panel-item {{ border: 1px solid #ddd; padding: 15px; background: white; text-align: center; border-radius: 5px; }}
                .panel-item img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
                .panel-info {{ margin-top: 10px; font-size: 14px; color: #666; }}
                button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
                button:hover {{ background: #0056b3; }}
                .download-btn {{ background: #28a745; }}
                .download-btn:hover {{ background: #218838; }}
                .back-btn {{ background: #6c757d; }}
                .back-btn:hover {{ background: #545b62; }}
                .summary {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Panel Extraction Results</h1>
            <a href="/"><button class="back-btn">← Upload Another Image</button></a>
            
            <div class="summary">
                <h3>Extraction Summary</h3>
                <p><strong>Original Image:</strong> {file.filename}</p>
                <p><strong>Panels Extracted:</strong> {len(panels)}</p>
                <p><strong>Output Directory:</strong> output/</p>
            </div>
        """
        
        if panels:
            html_content += f"""
            <div class="results">
                <h3>Extracted Panels</h3>
                <a href="/download-all/{Path(file.filename).stem}"><button class="download-btn">Download All Panels (ZIP)</button></a>
                
                <div class="panel-grid">
            """
            
            for panel in panels:
                html_content += f"""
                <div class="panel-item">
                    <img src="/output/{panel['filename']}" alt="Panel {panel['panel_number']}">
                    <div class="panel-info">
                        <strong>Panel {panel['panel_number']}</strong><br>
                        Size: {panel['dimensions']}<br>
                        Position: {panel['position']}<br>
                        <a href="/output/{panel['filename']}" download><button>Download</button></a>
                    </div>
                </div>
                """
            
            html_content += "</div></div>"
        else:
            html_content += """
            <div class="results">
                <h3>No Panels Detected</h3>
                <p>No panels were found in the uploaded image. This could be because:</p>
                <ul>
                    <li>The image doesn't contain comic panels</li>
                    <li>The panels are too small or unclear</li>
                    <li>The image needs better contrast</li>
                </ul>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Processing Image</h1>
            <p>Error: {str(e)}</p>
            <a href="/"><button>← Back to Upload</button></a>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)


@app.get("/download-all/{filename}")
async def download_all_panels(filename: str):
    """Download all panels as a ZIP file"""
    try:
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
                output_dir = Path("output")
                for panel_file in output_dir.glob(f"{filename}_panel_*.png"):
                    zip_file.write(panel_file, panel_file.name)
            
            return FileResponse(
                tmp_file.name,
                media_type='application/zip',
                filename=f"{filename}_panels.zip"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ZIP: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Panel Extractor Web Interface...")
    print("Open http://localhost:8002 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8002)