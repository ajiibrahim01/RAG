import os
from fastapi import FastAPI, BackgroundTasks
from main import process_markdown_files
from ocr import OCRProcessor

#INPUT_FOLDER = "/home/rocketji/RAG/Documents/input"
#OUTPUT_FOLDER = "/home/rocketji/RAG/Documents/output"
#MARKDOWN_FOLDER = r"E:\\CODE\\RAG\\Documents\\output"
#OUTPUT_FOLDER = r"E:\\CODE\\RAG\\Documents\\output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI(
    title="Image Description",
    description="Generate image captions and enrich markdown files using Qwen2.5-VL-7B."
)

ocr_processor = OCRProcessor(
    os.environ["OCR_PROCESS_URL"],
    os.environ["OCR_RETRIEVE_URL"]
)

def run_ocr_on_markdown_files(markdown_folder: str):
    """
    Run OCR on all markdown files in a given folder.
    """
    for filename in os.listdir(markdown_folder):
        if filename.endswith(".md"):
            markdown_file_path = os.path.join(markdown_folder, filename)
            print(f"Running OCR on {markdown_file_path}")
            ocr_processor.process_by_target(markdown_file_path, "all")


def run_ocr_pipeline(markdown_folder: str, output_folder: str):
    # VLM Processing
    print(f"Starting VLM processing from {markdown_folder} to {output_folder}")
    process_markdown_files(markdown_folder, output_folder)
    print("VLM processing complete.")
    
    #OCR Processing
    print(f"Starting OCR processing on files in {output_folder}")
    run_ocr_on_markdown_files(output_folder)
    print("OCR processing complete.")
    

@app.post("/image")
def process_markdowns(background_tasks: BackgroundTasks):
    """
    Run the markdown captioning pipeline.
    Folders are configured in code, not user input.
    """
    if not os.path.exists(MARKDOWN_FOLDER):
        return {"error": f"Markdown folder not found: {MARKDOWN_FOLDER}"}

    background_tasks.add_task(run_ocr_pipeline, MARKDOWN_FOLDER, OUTPUT_FOLDER)

    return {
        "status": "Processing started",
        "markdown_folder": MARKDOWN_FOLDER,
        "output_folder": OUTPUT_FOLDER,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=7000, reload=True)