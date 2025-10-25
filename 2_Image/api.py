import os
from fastapi import FastAPI, BackgroundTasks
from main import process_markdown_files

MARKDOWN_FOLDER = r"E:\\CODE\\RAG\\Documents\\output"
OUTPUT_FOLDER = r"E:\\CODE\\RAG\\Documents\\output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI(
    title="Image Description",
    description="Generate image captions and enrich markdown files using Qwen2.5-VL-7B."
)


@app.post("/image")
def process_markdowns(background_tasks: BackgroundTasks):
    """
    Run the markdown captioning pipeline.
    Folders are configured in code, not user input.
    """
    if not os.path.exists(MARKDOWN_FOLDER):
        return {"error": f"Markdown folder not found: {MARKDOWN_FOLDER}"}

    background_tasks.add_task(process_markdown_files, MARKDOWN_FOLDER, OUTPUT_FOLDER)

    return {
        "status": "Processing started",
        "markdown_folder": MARKDOWN_FOLDER,
        "output_folder": OUTPUT_FOLDER,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=7000, reload=True)