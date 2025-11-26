import re
import os
import requests
import json
import time

class OCRProcessor:
    def __init__(self, ocr_process_url, ocr_retrieve_base_url):
        self.process_url = ocr_process_url
        self.retrieve_base_url = ocr_retrieve_base_url
    
    def analyze_markdown(self, markdown_file):
        """Analyze markdown file and return detailed image information"""
        with open(markdown_file, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        images = []

        # Regex to find markdown images, allowing for newlines between alt text and path
        img_pattern = r'!\[([^\]]*)\]\s*\(([^)]+\.(png|jpg|jpeg|gif|webp|svg))\)'
        
        for match in re.finditer(img_pattern, content, re.IGNORECASE):
            # To find the line number, we count newlines before the match start
            line_num = content.count('\n', 0, match.start()) + 1
            
            # Create unique identifier
            img_id = f"line_{line_num}_pos_{match.start()}"
            
            # Get context
            start_context = max(0, line_num - 2)
            end_context = min(len(lines), line_num + 1)
            context = lines[start_context:end_context]
            
            images.append({
                'id': img_id,
                'line_number': line_num,
                'position': match.start(), # This position is in the whole content, not the line
                'alt_text': match.group(1).strip(),
                'filepath': match.group(2).strip(),
                'filename': os.path.basename(match.group(2).strip()),
                'full_line': match.group(0).replace('\n', ' '), # a representation of the match
                'context': context,
                'raw_match': match.group(0)
            })
            
        return images
    
    def show_image_report(self, markdown_file):
        """Display comprehensive report of all images found"""
        images = self.analyze_markdown(markdown_file)
        
        print(f"\n Image Analysis Report for {markdown_file}")
        print("=" * 80)
        print(f"Found {len(images)} image(s):\n")
        
        for i, img in enumerate(images, 1):
            print(f"{i}. {img['filename']}")
            print(f"Location: Line {img['line_number']}, Position {img['position']}")
            print(f"Alt Text: '{img['alt_text']}'")
            print(f"Full Path: {img['filepath']}")
            print(f"Unique ID: {img['id']}")
            print(f"Full Line: {img['full_line']}")
            
            # Show context
            print(f"Context:")
            context_lines = []
            for j, ctx_line in enumerate(img['context']):
                if j == 1:  # The target line
                    context_lines.append(f"   >>> {ctx_line}")
                else:
                    context_lines.append(f"       {ctx_line}")
            print('\n'.join(context_lines))
            print()
        
        return images
    
    def process_by_target(self, markdown_file, target_specifier):
        """
        Process images based on precise targeting
        
        Target formats:
        - "line_N": Nth image in document (line_1, line_2, etc.)
        - "filepath:filename.png": All images with this filepath
        - "line_number:N": Image on specific line N
        - "id:unique_id": Specific image by unique ID
        - "all": Process all images
        """
        all_images = self.analyze_markdown(markdown_file)
        
        if not all_images:
            print("No images found in markdown file.")
            return False
        
        # Determine target images
        target_images = self._resolve_target(all_images, target_specifier)
        
        if not target_images:
            print(f"No images found matching target: {target_specifier}")
            return False
        
        print(f"Target: {target_specifier}")
        print(f"Will process {len(target_images)} image(s):")
        for img in target_images:
            print(f"   - Line {img['line_number']}: {img['filename']}")
        print()
        
        # Process the images
        return self._process_images(markdown_file, target_images)
    
    def _resolve_target(self, all_images, target_specifier):
        """Resolve target specifier to list of images"""
        
        if target_specifier == "all":
            return all_images
        
        elif target_specifier.startswith("line_"):
            try:
                image_index = int(target_specifier.split('_')[1]) - 1
                if 0 <= image_index < len(all_images):
                    return [all_images[image_index]]
            except (ValueError, IndexError):
                pass
                
        elif target_specifier.startswith("filepath:"):
            filepath = target_specifier.split(':', 1)[1]
            return [img for img in all_images if img['filepath'] == filepath]
            
        elif target_specifier.startswith("line_number:"):
            try:
                line_num = int(target_specifier.split(':', 1)[1])
                return [img for img in all_images if img['line_number'] == line_num]
            except ValueError:
                pass
                
        elif target_specifier.startswith("id:"):
            img_id = target_specifier.split(':', 1)[1]
            return [img for img in all_images if img['id'] == img_id]
        
        return []
    
    def _process_images(self, markdown_file, target_images):
        """Process the specified images"""
        try:
            with open(markdown_file, 'r') as f:
                lines = f.readlines()
            
            processed_count = 0
            
            # Process in reverse order to maintain line numbers
            for img in reversed(target_images):
                line_index = img['line_number'] - 1
                
                # Check if OCR already exists
                if line_index + 1 < len(lines):
                    next_line = lines[line_index + 1]
                    if "### OCR Results:" in next_line:
                        print(f"Skipping {img['filename']} (line {img['line_number']}) - OCR results already exist")
                        continue
                
                # Check if image file exists
                full_image_path = os.path.join(os.path.dirname(markdown_file), img['filepath'])
                if not os.path.exists(full_image_path):
                    print(f"⚠️  Image file not found: {full_image_path}")
                    continue
                
                print(f"Processing: {img['filename']} (line {img['line_number']})")
                
                # OCR Processing
                ocr_result = self._perform_ocr(full_image_path, img['filename'])
                
                if ocr_result:
                    # Insert OCR content
                    lines.insert(line_index + 1, ocr_result)
                    processed_count += 1
                else:
                    print(f"Failed to process {img['filename']}")
            
            # Write results
            if processed_count > 0:
                with open(markdown_file, 'w') as f:
                    f.writelines(lines)
                print(f"\nUpdated {markdown_file} with OCR results for {processed_count} image(s)")
                return True
            else:
                print("\nNo new images processed")
                return False
                
        except Exception as e:
            print(f"Error processing file: {e}")
            return False
    
    def _perform_ocr(self, image_path, filename, max_attempts=30):
        """Perform OCR on a single image"""
        try:
            # Upload for processing
            with open(image_path, 'rb') as file:
                files = {'file': (filename, file, 'image/png')}
                headers = {'Accept': 'application/json'}
                response = requests.post(self.process_url, files=files, headers=headers)
            
            if response.status_code not in [200, 201]:
                print(f"Upload failed: {response.status_code}")
                return None
            
            response_data = response.json()
            document_id = response_data['data']['documentId']
            
            # Poll for results
            retrieve_url = f"{self.retrieve_base_url}/{document_id}"
            
            for attempt in range(max_attempts):
                retrieve_response = requests.get(retrieve_url)
                retrieve_data = retrieve_response.json()
                
                if retrieve_response.status_code == 200:
                    status = retrieve_data['data'].get('status', '')
                    
                    if status == 'COMPLETED':
                        print(f"✅ OCR completed for {filename}")
                        return self._format_ocr_results(retrieve_data)
                        
                    elif status == 'PROCESSING':
                        print(f"⏳ Processing {filename}... (attempt {attempt + 1})")
                        time.sleep(2)
                    else:
                        print(f"Processing failed: {status}")
                        return None
                else:
                    print(f"Retrieval error: {retrieve_response.status_code}")
                    return None
            
            print(f"Timeout processing {filename}")
            return None
            
        except Exception as e:
            print(f"OCR error for {filename}: {e}")
            return None

    def _format_ocr_results(self, retrieve_data):
        """Format OCR results for markdown insertion"""
        extraction_result = retrieve_data['data'].get('extractionResult', {})
        pages = extraction_result.get('pages', [])

        if not pages:
            return None

        ocr_content = f"\n*OCR Result:* "

        for page in pages:

            res = page.get('res', {})
            overall_ocr = res.get('overall_ocr_res', {})
            rec_texts = overall_ocr.get('rec_texts', [])


            if rec_texts:
                ocr_content += f"{rec_texts}\n"
            ocr_content += "\n"  # Add a newline for separation between pages

        return ocr_content

if __name__ == "__main__":
    # Initialize processor with your OCR endpoints
    processor = OCRProcessor(
        os.environ["OCR_PROCESS_URL"],
        os.environ["OCR_RETRIEVE_URL"]
    )
    
    # Example usage
    markdown_file = "test_fresh.md"
    
    # Step 1: Process all images in the markdown file
    processor.process_by_target(markdown_file, "all")