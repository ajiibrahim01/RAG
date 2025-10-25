import os
import re
import base64
from openai import OpenAI
from dotenv import load_dotenv

# vlm used by hf inference provider :  Qwen/Qwen2.5-VL-7B-Instruct

load_dotenv("E:\CODE\RAG\.env")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN_VLM"]
)

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def extract_images_and_context(markdown_path):
    """Extract image paths and surrounding context from a markdown file."""
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    image_data = []
    for i, line in enumerate(lines):
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            img_filename = os.path.basename(img_path)
            context_before = " ".join(lines[max(0, i - 2):i]).strip()
            context_after = " ".join(lines[i + 1:min(len(lines), i + 3)]).strip()
            image_data.append((img_filename, context_before, context_after))
    return image_data, lines


def generate_caption_api(image_path, context_before, context_after):
    """Generate an image caption using Hugging Face inference provider."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return "[Image description unavailable]"

    # Encode image as a data URL (Hugging Face supports base64-embedded images)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    prompt_text = f"Context: {context_before} ... {context_after}. Please describe the image briefly in one paragraph."

    print(f"Generating caption for: {os.path.basename(image_path)} ...")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            max_tokens=256,
        )

        # The message may be dict-like (OpenAI client can vary slightly)
        message = response.choices[0].message
        if isinstance(message, dict):
            caption = message.get("content", "").strip()
        else:
            caption = message.content.strip()

        print(f"Caption done: {os.path.basename(image_path)}")
        return caption or "[Empty description]"

    except Exception as e:
        print(f"Error while calling API for {os.path.basename(image_path)}: {e}")
        return "[Image description failed]"


def update_markdown(markdown_path, image_data, lines, output_folder):
    """Insert generated captions into the markdown output."""
    new_lines = []
    for line in lines:
        new_lines.append(line)
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            img_filename = os.path.basename(img_path)
            caption = next(
                (desc for img, _, _, desc in image_data if img == img_filename),
                "[Image description unavailable]",
            )
            new_lines.append(f"\n*Image Description:* {caption}\n")

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(markdown_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"✅ Saved updated markdown: {output_path}")


def process_markdown_files(markdown_folder, output_folder):
    """Process all markdown files in the given folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = [f for f in os.listdir(markdown_folder) if f.endswith(".md")]
    print(f"Found {len(all_files)} markdown file(s) in folder '{markdown_folder}'")

    for file_idx, file in enumerate(all_files, start=1):
        print(f"\n🔹 Processing file {file_idx}/{len(all_files)}: {file}")
        markdown_path = os.path.join(markdown_folder, file)
        filename_without_ext = os.path.splitext(file)[0]
        image_folder = os.path.join(markdown_folder, f"{filename_without_ext}_artifacts")

        if not os.path.exists(image_folder):
            print(f"⚠️ Warning: Image folder '{image_folder}' not found for '{file}'")
            continue

        image_data, lines = extract_images_and_context(markdown_path)
        print(f"Found {len(image_data)} image(s) in {file}")

        enriched_data = []
        for img_idx, (img_filename, context_before, context_after) in enumerate(image_data, start=1):
            print(f" [{img_idx}/{len(image_data)}] Captioning {img_filename} ...")
            full_image_path = os.path.join(image_folder, img_filename)
            caption = generate_caption_api(full_image_path, context_before, context_after)
            enriched_data.append((img_filename, context_before, context_after, caption))

        update_markdown(markdown_path, enriched_data, lines, output_folder)
        print(f"Finished processing: {file}")


# if __name__ == "__main__":
#     markdown_folder = r"E:\\CODE\\RAG\\Documents\\output"
#     output_folder = r"E:\\CODE\\RAG\\Documents\\output"

#     process_markdown_files(markdown_folder, output_folder)