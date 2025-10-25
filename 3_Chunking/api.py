from fastapi import FastAPI
import os
import main

app = FastAPI()

# Set your fixed input and output directories
INPUT_FOLDER = r"E:\\CODE\\RAG\\Documents\\output"
OUTPUT_FOLDER = r"E:\\CODE\\RAG\\Documents\\chunk"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/chunking")
def chunk_all_markdown_files():
    """Process all markdown files in input_md and save the results to output folder."""
    processed_files = []
    
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(".md"):
            input_path = os.path.join(INPUT_FOLDER, file_name)
            # Let main.py use its own default output logic
            main.process_markdown(input_path, OUTPUT_FOLDER)
            processed_files.append(file_name)

    return {
        "message": "✅ Processing completed.",
        "processed_files": processed_files,
        "output_dir": OUTPUT_FOLDER
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)

