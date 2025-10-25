from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
import shutil
from pathlib import Path
import logging
from main import convert_and_save 

app = FastAPI()

# INPUT_DIR = r"E:\\CODE\\RAG\\Documents" 
OUTPUT_DIR = r"E:\\CODE\\RAG\\Documents\\output"   # Folder for outputs

@app.post("/parsing")
async def upload_and_process(file: UploadFile = File(...)):
    """Upload a file and convert it to Markdown."""
    try:
        # Save uploaded file temporarily
        temp_path = Path(OUTPUT_DIR) / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run conversion on uploaded file
        result = convert_and_save(temp_path, Path(OUTPUT_DIR))

        if result["success"]:
            return {"message": result["message"], "converted_files": result.get("files", [])}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)