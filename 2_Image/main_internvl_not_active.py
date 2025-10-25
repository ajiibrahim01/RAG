import os
import re
import base64
from openai import OpenAI


# InternVL
client = OpenAI(
    api_key="sk-dZr32cb1MKGvHaxKQsvQZnFoAJEHOamwJxNguuo5qwvOWUEn",
    base_url="https://chat.intern-ai.org.cn/api/v1/"
)

def extract_images_and_context(markdown_path):
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    image_data = []
    for i, line in enumerate(lines):
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            img_filename = os.path.basename(img_path)
            context_before = " ".join(lines[max(0, i-2):i]).strip()
            context_after = " ".join(lines[i+1:min(len(lines), i+3)]).strip()
            image_data.append((img_filename, context_before, context_after))
    return image_data, lines

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

import math

def generate_caption_api(image_path, context_before, context_after):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return "[Image description unavailable]"

    base64_image = encode_image(image_path)
    prompt = f"Context: {context_before} ... {context_after}. Please describe the image shortly.\n<image>{base64_image}</image>"

    # ---- Token estimation ----
    base64_length = len(base64_image)
    text_length = len(prompt) - base64_length
    estimated_tokens_image = math.ceil(base64_length / 4)  # ~1 token per 4 chars
    estimated_tokens_text = math.ceil(text_length / 4)     # rough text tokens
    estimated_total_tokens = estimated_tokens_image + estimated_tokens_text

    print(f"Generating caption for: {os.path.basename(image_path)} ...")
    print(f"   Estimated input tokens: {estimated_total_tokens:,} "
          f"(Image ≈ {estimated_tokens_image:,}, Text ≈ {estimated_tokens_text:,})")

    # Optional: Warn if near typical limits
    if estimated_total_tokens > 100_000:
        print("⚠️ Warning: This image prompt is very large and may exceed API token limits.")

    try:
        response = client.chat.completions.create(
            model="internvl3.5-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        caption = response.choices[0].message.content.strip()
        print(f"Caption done: {os.path.basename(image_path)}")
        return caption

    except Exception as e:
        print(f"Error while calling API for {os.path.basename(image_path)}: {e}")
        return "[Image description failed]"


def process_markdown_files(markdown_folder, output_folder):
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
            print(f"Warning: Image folder '{image_folder}' not found for '{file}'")
            continue

        image_data, lines = extract_images_and_context(markdown_path)
        print(f"Found {len(image_data)} image(s) in {file}")

        enriched_data = []
        for img_idx, (img_filename, context_before, context_after) in enumerate(image_data, start=1):
            print(f"   [{img_idx}/{len(image_data)}] Captioning {img_filename} ...")
            full_image_path = os.path.join(image_folder, img_filename)
            caption = generate_caption_api(full_image_path, context_before, context_after)
            enriched_data.append((img_filename, context_before, context_after, caption))

        update_markdown(markdown_path, enriched_data, lines, output_folder)
        print(f"Finished processing: {file}")


def update_markdown(markdown_path, image_data, lines, output_folder):
    new_lines = []
    for line in lines:
        new_lines.append(line)
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            img_filename = os.path.basename(img_path)
            caption = next((desc for img, _, _, desc in image_data if img == img_filename), "[Image description unavailable]")
            new_lines.append(f"\n*Image Description:* {caption}\n")

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, os.path.basename(markdown_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    markdown_folder = r"E:\\CODE\\RAG\\Documents\\output"
    output_folder = r"E:\\CODE\\RAG\\Documents\\output"
    process_markdown_files(markdown_folder, output_folder)
