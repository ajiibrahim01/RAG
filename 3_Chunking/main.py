import json
import re
import os
from pathlib import Path

# === Logic Functions ===
def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def is_table(chunk):
    return bool(re.search(r'^\|.*\|\n\|[-| ]+\|\n(\|.*\|\n)*', chunk, re.MULTILINE))

def extract_and_split_table(chunk, max_rows=10):
    lines = chunk.strip().split("\n")
    header, table_rows = None, []
    for i, line in enumerate(lines):
        if re.match(r'^\|[-| ]+\|$', line):
            header = lines[i - 1].strip("|").split("|")
            header = [h.strip() for h in header]
            continue
        if header:
            row_data = line.strip("|").split("|")
            row_data = [cell.strip() for cell in row_data]
            table_rows.append(row_data)
    table_chunks = [
        {"headers": header, "rows": table_rows[i:i + max_rows]}
        for i in range(0, len(table_rows), max_rows)
    ]
    return table_chunks if header and table_rows else None

def extract_section_title(header):
    match = re.match(r'^(#+)\s+(.*)', header.strip())
    return match.group(2) if match else None

def detect_table_title(pre_table_text):
    lines = pre_table_text.strip().split("\n")
    return lines[-1] if lines and len(lines[-1].split()) < 10 else None

def split_text(text, section_title, max_words=400, overlap=40):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        if start == 0:
            chunk = f"## {section_title}\n{chunk}"
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

def clean_dot_leaders(text):
    return re.sub(r'\.{2,}', '', text).strip()

def is_table_of_contents(table_rows):
    score = 0
    for row in table_rows:
        row_text = " ".join(row)
        if re.search(r'\.{5,}', row_text) and re.search(r'[A-Za-z]', row_text):
            score += 1
    return score >= len(table_rows) * 0.6

def structure_toc(table_rows, current_section):
    toc = []
    for row in table_rows:
        if len(row) == 1:
            title = clean_dot_leaders(row[0])
            toc.append({"number": None, "title": title, "section": current_section})
        elif len(row) >= 2:
            number = clean_dot_leaders(row[0])
            title = clean_dot_leaders(row[1])
            toc.append({"number": number, "title": title, "section": current_section})
    return toc

def autofill_toc_numbers(toc_items):
    counter = 1
    for item in toc_items:
        if not item.get("number") or item["number"] is None:
            item["number"] = str(counter)
            counter += 1
    return toc_items

def process_markdown(file_path, output_dir="output_json"):
    markdown_text = load_markdown(file_path)
    file_name_only = os.path.basename(file_path)
    sections = re.split(r'^(#+\s+.*)', markdown_text, flags=re.MULTILINE)
    final_chunks, toc_items, current_section, chunk_id = [], [], "Unknown", 1

    def append_chunk(content, section, position):
        final_chunks.append({
            "chunk_id": position,
            "content": content.strip(),
            "metadata": {
                "source": file_name_only,
                "section": section,
                "position": position
            }
        })

    # Preamble section
    if sections[0].strip():
        preamble = sections[0].strip()
        preamble_chunks = split_text(preamble, "Preamble")
        for chunk in preamble_chunks:
            append_chunk(chunk, "Preamble", chunk_id)
            chunk_id += 1

    for i in range(1, len(sections), 2):
        section_title = extract_section_title(sections[i]) or current_section
        content = sections[i + 1].strip()
        current_section = section_title

        # Handle TOC and tables
        table_matches = list(re.finditer(r'(\|.*\|\n\|[-| ]+\|\n(?:\|.*\|\n)+)', content, re.MULTILINE))
        last_index = 0

        for match in table_matches:
            start, end = match.span()
            pre_table_text = content[last_index:start].strip()
            table_text = match.group(0)
            last_index = end

            grouped = group_images_with_text(pre_table_text)
            for text_block in grouped:
                text_chunks = split_text(text_block, section_title)
                for chunk in text_chunks:
                    append_chunk(chunk, section_title, chunk_id)
                    chunk_id += 1

            table_chunks = extract_and_split_table(table_text)
            if table_chunks:
                headers = table_chunks[0]["headers"]
                rows_flat = sum((tc["rows"] for tc in table_chunks), [])
                if is_table_of_contents(rows_flat):
                    toc_items.extend(structure_toc(rows_flat, section_title))
                else:
                    table_title = detect_table_title(pre_table_text)
                    for table_chunk in table_chunks:
                        final_chunks.append({
                            "chunk_id": chunk_id,
                            "table": table_chunk,
                            "metadata": {
                                "source": file_name_only,
                                "section": section_title,
                                "table_title": table_title,
                                "position": chunk_id
                            }
                        })
                        chunk_id += 1

        remaining_text = content[last_index:].strip()
        grouped = group_images_with_text(remaining_text)
        for text_block in grouped:
            text_chunks = split_text(text_block, section_title)
            for chunk in text_chunks:
                append_chunk(chunk, section_title, chunk_id)
                chunk_id += 1

    toc_items = autofill_toc_numbers(toc_items)
    if toc_items:
        final_chunks.insert(0, {
            "chunk_id": 0,
            "toc_items": toc_items,
            "metadata": {"source": file_name_only, "section": "ToC", "position": 0}
        })

    # Use the provided output directory
    output_path = Path(output_dir) / file_name_only.replace(".md", ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=4, ensure_ascii=False)
    print(f"✅ Chunking complete. Saved to {output_path}")



def group_images_with_text(content):
    """Combines image tags with adjacent descriptions or paragraphs."""
    image_pattern = r'(!\[.*?\]\(.*?\))((?:\s+\*Image Description:\*.*)?)'
    blocks = []
    last = 0
    for match in re.finditer(image_pattern, content, re.DOTALL):
        start, end = match.span()
        prefix = content[last:start].strip()
        if prefix:
            blocks.append(prefix)
        blocks.append(content[start:end].strip())
        last = end
    suffix = content[last:].strip()
    if suffix:
        blocks.append(suffix)
    return blocks
