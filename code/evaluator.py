# Standard libraries
from openai import OpenAI
import os
import re
import logging
import base64
from datetime import datetime
import json
from typing import List, Dict, Any

# PDF and image processing
import PyPDF2
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='slide_review.log')

# OpenAI API configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configure Tesseract path
if os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# Review prompt template
REVIEW_PROMPT = """
Consider the learning materials represented as a pdf file attached and
put together a critical evaluation along the lines of the criteria below:

Consistency and Alignment
- Do the slides align with the Learning Outcomes and module handbook description in terms of content,
  depth, and focus?
- Do the assessments (labs, CA, and exams) reflect the skills and knowledge outlined in the module
  learning outcomes?
- Are key concepts and skills introduced progressively across the module in a way that supports
  student learning?

Assessment and Lab Effectiveness
- Do the labs effectively prepare students for their project work and support their understanding of
  key concepts?
- Do the exams and CA build on each other to provide a structured approach to learning and assessment?
- Are the types of questions in the exams reflective of the skills students need for the project and
  continuous assessments?

Clarity and Understanding
- Are the slides structured logically, ensuring that concepts build on each other in a way that
  supports understanding?
- Is there unnecessary repetition in the slides that could be streamlined?
- Are the explanations clear and appropriate for the students' level, avoiding excessive
  complexity or oversimplification?
- Are key concepts illustrated effectively through examples, diagrams, or case studies?
- Are important terms, formulas, and concepts adequately explained with definitions and
  context?

Accuracy and Completeness
- Are the slides factually accurate, avoiding outdated or incorrect information?
- Are there any gaps in the slides where additional explanation or context would
  improve understanding?
- Are important distinctions and nuances in the subject matter properly addressed?

Engagement and Effectiveness of Delivery
- Do the slides incorporate elements that make them engaging and interactive (e.g.,
  thought-provoking questions, activities, real-world applications)?
- Are technical concepts explained in a way that encourages engagement and critical
  thinking rather than passive memorization?
- Is there sufficient scaffolding for difficult topics, ensuring that students have
  the necessary background before encountering advanced concepts?

Synchronous and Asynchronous Learning
- Are the slides suitable for synchronous learning?
- Do the slides encourage and include offerings for asynchronous learning?

Critique and Improvements
- Based on the analysis of the slides, what improvements can be made to enhance clarity,
  engagement, and effectiveness?
- Are there better ways to organize or structure the content to support student learning?
- Are there any missing topics or underexplored areas that should be covered in greater
  depth?

Avoid general statements like "Could profit from better scaffolding". Instead, offer
specific suggestions on what can actually be done to improve the slides on the point
flagged up.

In the summary, highlight the 3 areas that most urgently need attention.

Slide Content:
{slide_text}
"""

def clean_analysis_output(analysis: str) -> str:
    """Remove empty sections from the analysis."""
    sections = re.split(r'(\\section{.*?}|\\subsection{.*?})', analysis)
    cleaned_sections = []
    current_section = []

    for section in sections:
        if section.strip():
            if section.startswith('\\section') or section.startswith('\\subsection'):
                if current_section:
                    cleaned_sections.extend(current_section)
                current_section = [section]
            else:
                content = section.strip()
                if content and not re.match(r'^\\begin{itemize}\s*\\end{itemize}\s*$', content):
                    current_section.append(content)

    if current_section and len(current_section) > 1:
        cleaned_sections.extend(current_section)

    return '\n'.join(cleaned_sections)

def escape_latex_chars(text: str) -> str:
    """Escape LaTeX special characters."""
    escape_chars = {
        '%': '\\%',
        '#': '\\#',
        '_': '\\_'
    }
    for char, escape in escape_chars.items():
        text = text.replace(char, escape)
    return text

def clean_latex_content(content: str) -> str:
    """
    Clean LaTeX content to ensure proper list environments and restrict commands.
    Prevents nested itemize environments and ensures proper closure.
    """
    def convert_markdown_headers(match):
        hashes = match.group(1)
        title = match.group(2).strip()
        if len(hashes) == 3:
            return f"\\subsection{{{title}}}"
        elif len(hashes) in [1, 2]:
            return f"\\section{{{title}}}"
        return title

    content = re.sub(r'^(#{1,3})\s*(.+?)$', convert_markdown_headers, content, flags=re.MULTILINE)
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)

    def flatten_itemize(text):
        while '\\begin{itemize}\\begin{itemize}' in text:
            text = text.replace('\\begin{itemize}\\begin{itemize}', '\\begin{itemize}')
        while '\\end{itemize}\\end{itemize}' in text:
            text = text.replace('\\end{itemize}\\end{itemize}', '\\end{itemize}')
        return text

    lines = content.split('\n')
    processed_lines = []
    in_itemize = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if '\\begin{itemize}' in line:
            if not in_itemize:
                processed_lines.append('\\begin{itemize}')
                in_itemize = True
            continue

        if '\\end{itemize}' in line:
            if in_itemize:
                processed_lines.append('\\end{itemize}')
                in_itemize = False
            continue

        if '\\item' in line:
            if not in_itemize:
                processed_lines.append('\\begin{itemize}')
                in_itemize = True
            processed_lines.append(line)
            continue

        if line.startswith('\\section{') or line.startswith('\\subsection{') or line.startswith('\\chapter{'):
            if in_itemize:
                processed_lines.append('\\end{itemize}')
                in_itemize = False
            processed_lines.append(line)
            continue

        if in_itemize and not line.startswith('\\'):
            if processed_lines and '\\item' in processed_lines[-1]:
                processed_lines[-1] = processed_lines[-1] + ' ' + line
            continue
        elif not in_itemize and not line.startswith('\\'):
            processed_lines.append(line)

    if in_itemize:
        processed_lines.append('\\end{itemize}')

    content = '\n'.join(processed_lines)
    content = flatten_itemize(content)
    content = re.sub(r'\\begin{itemize}\s*\\end{itemize}', '', content)
    content = re.sub(r'\n\s*\n', '\n\n', content)

    return content.strip()

def generate_content_analysis(content_text: str) -> str:
    """Generate the content-related analysis using GPT-4."""
    try:
        system_prompt = """You are an expert educational content reviewer analyzing course materials.
Generate a structured LaTeX analysis focusing on pedagogical effectiveness and content organization.
Use sections and subsections to organize your analysis. Focus on:
1. Consistency and alignment with learning outcomes
2. Progressive introduction of concepts
3. Assessment effectiveness
4. Clarity and understanding
5. Content accuracy and completeness
Be specific and provide concrete examples from the content."""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": REVIEW_PROMPT.format(slide_text=content_text)}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        return clean_latex_content(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error generating content analysis: {e}")
        return ""

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from all PDF pages efficiently."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            total_pages = len(reader.pages)
            for i, page in enumerate(reader.pages, 1):
                text.append(page.extract_text())
                if i % 10 == 0:
                    print(f"  Extracted text from {i}/{total_pages} pages")
            return "\n".join(text)
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def format_slide_analysis(analyses: List[Dict[str, Any]]) -> str:
    """Format the visual analysis section as a table EXACTLY as shown."""
    if not analyses:
        return ""

    # Start with section header
    result = ["\\section{Visual Analysis}"]

    # Begin table - NO escaping of ampersands
    table_lines = []
    table_lines.append("\\begin{tabular}{|l|c|c|}")
    table_lines.append("\\hline")
    table_lines.append("Slide Number & Text Area & Visual Elements \\\\")
    table_lines.append("\\hline")

    # Add each slide's data
    for analysis in analyses:
        if analysis:
            slide_number = analysis.get('page_number', 0)
            text_ratio = analysis.get('text_area_ratio', 0) * 100
            visual_ratio = analysis.get('visual_area_ratio', 0) * 100

            table_lines.append(f"Slide {slide_number} & {text_ratio:.1f}\\% & {visual_ratio:.1f}\\% \\\\")
            table_lines.append("\\hline")

    # Close table
    table_lines.append("\\end{tabular}")

    # Join all lines with newlines
    result.extend(table_lines)
    return '\n'.join(result)

def analyze_slide_content(image_path: str, page_number: int) -> Dict[str, Any]:
    """Analyze content and visual elements of a single slide."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {}

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create masks for text and visual elements
        text_mask = np.zeros(gray.shape, dtype=np.uint8)

        # Use Tesseract for text detection
        text_data = pytesseract.image_to_data(Image.fromarray(img), output_type=pytesseract.Output.DICT)
        for i, text in enumerate(text_data['text']):
            if text.strip():
                x, y, w, h = (text_data['left'][i], text_data['top'][i],
                           text_data['width'][i], text_data['height'][i])
                cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)

        # Detect visual elements
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        visual_mask = np.zeros(gray.shape, dtype=np.uint8)
        min_area = (width * height) * 0.05
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(visual_mask, [cnt], -1, 255, -1)

        # Calculate actual ratios
        total_pixels = width * height
        text_ratio = np.count_nonzero(text_mask) / total_pixels
        visual_ratio = np.count_nonzero(visual_mask) / total_pixels

        return {
            'page_number': page_number,
            'text_area_ratio': text_ratio,
            'visual_area_ratio': visual_ratio
        }
    except Exception as e:
        logging.error(f"Error analyzing slide {page_number}: {e}")
        return {}

def pdf_to_images(pdf_path):
    """Convert PDF pages to images."""
    try:
        # Create a temporary directory for images
        os.makedirs('temp_pdf_images', exist_ok=True)

        # Get total number of pages
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            total_pages = len(pdf.pages)

        print(f"Converting {total_pages} pages to images...")
        # Convert PDF to images (all pages)
        images = convert_from_path(pdf_path, first_page=1, last_page=total_pages)

        image_paths = []
        for i, image in enumerate(images):
            image_path = f'temp_pdf_images/page_{i+1}.png'
            image.save(image_path, 'PNG')
            image_paths.append(image_path)

        return image_paths
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return []

def extract_slide_title(img: np.ndarray, page_number: int) -> str:
    """Extract slide title focusing on top left/center area."""
    try:
        height, width = img.shape[:2]

        top_height = int(height * 0.2)
        left_width = int(width * 0.4)
        center_start = int(width * 0.3)
        center_width = int(width * 0.4)

        left_region = img[0:top_height, 0:left_width]
        left_text = pytesseract.image_to_string(Image.fromarray(left_region))

        center_region = img[0:top_height, center_start:center_start+center_width]
        center_text = pytesseract.image_to_string(Image.fromarray(center_region))

        title_candidates = []

        left_lines = [line.strip() for line in left_text.split('\n') if line.strip()]
        if left_lines:
            title_candidates.extend(left_lines)

        center_lines = [line.strip() for line in center_text.split('\n') if line.strip()]
        if center_lines:
            title_candidates.extend(center_lines)

        filtered_candidates = [
            t for t in title_candidates
            if len(t) > 3 and len(t) < 100
            and not any(x in t.lower() for x in ['page', 'slide', 'digital4business', 'university'])
        ]

        if filtered_candidates:
            return filtered_candidates[0]

        return f"Slide {page_number}"

    except Exception as e:
        logging.error(f"Error extracting title for slide {page_number}: {e}")
        return f"Slide {page_number}"

def batch_analyze_slides(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple slides in batch."""
    analyses = []
    total_slides = len(image_paths)

    print(f"\nAnalyzing {total_slides} slides...")
    for i, image_path in enumerate(image_paths, 1):
        if i % 5 == 0:
            print(f"  Processed {i}/{total_slides} slides")
        analysis = analyze_slide_content(image_path, page_number=i)
        if analysis:
            analyses.append(analysis)

    return analyses

def generate_overall_analysis(slide_analyses: List[Dict[str, Any]], content_text: str) -> str:
    """Generate comprehensive analysis including both content and visual aspects."""
    try:
        # Generate content analysis
        print("Generating content analysis...")
        content_analysis = generate_content_analysis(content_text)

        # Generate visual analysis
        print("Generating visual analysis...")
        valid_analyses = [a for a in slide_analyses if a]
        visual_analysis = format_slide_analysis(valid_analyses) if valid_analyses else ""

        # Combine analyses
        combined_analysis = []

        # Add content analysis if present
        if content_analysis.strip():
            combined_analysis.append(content_analysis)

        # Add visual analysis if present
        if visual_analysis.strip():
            combined_analysis.append(visual_analysis)

        # Join with double newlines for clear separation
        final_analysis = "\n\n".join(combined_analysis)

        # Clean up empty sections
        return clean_analysis_output(final_analysis)

    except Exception as e:
        logging.error(f"Error generating overall analysis: {e}")
        return ""

def batch_review_slides(module_files: Dict[str, List[str]], output_folder: str) -> None:
    """Process all slides in batches, minimizing API calls."""
    os.makedirs(output_folder, exist_ok=True)
    total_modules = len(module_files)

    for module_index, (module_name, pdf_files) in enumerate(module_files.items(), 1):
        print(f"\nProcessing module {module_index}/{total_modules}: {module_name}")
        module_output = os.path.join(output_folder, module_name)
        os.makedirs(module_output, exist_ok=True)

        for pdf_index, pdf_path in enumerate(pdf_files, 1):
            try:
                filename = os.path.basename(pdf_path)
                print(f"\nProcessing file {pdf_index}/{len(pdf_files)}: {filename}")

                content_text = extract_pdf_text(pdf_path)
                image_paths = pdf_to_images(pdf_path)

                if not image_paths:
                    continue

                slide_analyses = batch_analyze_slides(image_paths)
                analysis = generate_overall_analysis(slide_analyses, content_text)

                output_path = os.path.join(module_output, f"{os.path.splitext(filename)[0]}.tex")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"\\chapter{{{os.path.splitext(filename)[0]}}}\n")
                    f.write(analysis)

                print(f"  Analysis completed and saved to {output_path}")

            except Exception as e:
                logging.error(f"Error processing {filename} in module {module_name}: {e}")
                print(f"  Error processing file: {e}")

def create_latex_book(output_folder: str, book_path: str) -> None:
    """Create final LaTeX book with all analyses."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    book_content = """\\documentclass{book}
\\usepackage[margin=1in]{geometry}
\\usepackage{datetime}
\\usepackage[colorlinks=true,
            linkcolor=blue,
            urlcolor=blue,
            bookmarks=true,
            bookmarksopen=true]{hyperref}
\\title{Digital4Business -- AI-based Slide Evaluations}
\\author{Dietmar Janetzko}
\\date{\\vspace{1em}Generated on: %s}
\\begin{document}
\\maketitle

\\chapter*{Preface}
This book offers evaluations of the course material of Digital4Business Master's Course. The evaluations focus on both content and visual aspects of the teaching materials, providing a comprehensive analysis of their pedagogical effectiveness.

All evaluations have been performed using ChatGPT 4.0, following specific criteria that can be found in Chapter 1. The evaluation process combines automated analysis of visual elements with pedagogical assessment, ensuring a thorough review of both content structure and presentation effectiveness.

The analysis covers multiple aspects:
\\begin{itemize}
\\item Content alignment with learning objectives
\\item Assessment and laboratory effectiveness
\\item Clarity and understanding of presented materials
\\item Accuracy and completeness of content
\\item Visual design effectiveness for learning
\\item Suitability for both synchronous and asynchronous learning
\\end{itemize}

This automated evaluation system aims to support continuous improvement in course material quality, ensuring optimal learning experiences for students in the Digital4Business Master's Course.
Each module's materials are evaluated independently, providing specific recommendations for potential improvements while highlighting effective teaching approaches already in place.
The code used in this report can be found under \\href{https://github.com/dietmarja/slide\_evaluator}{https://github.com/dietmarja/slide\_evaluator}.




\\tableofcontents

\\chapter{Evaluation Criteria}
\\begin{verbatim}
%s
\\end{verbatim}
""" % (timestamp, REVIEW_PROMPT.replace("Slide Content:\n{slide_text}", "").replace("&", "\\&"))

    for module_name in sorted(os.listdir(output_folder)):
        module_path = os.path.join(output_folder, module_name)
        if os.path.isdir(module_path):
            book_content += f"\n\\part{{Module {module_name}}}\n"
            tex_files = sorted([f for f in os.listdir(module_path) if f.endswith('.tex')])
            for tex_file in tex_files:
                file_name_without_ext = os.path.splitext(tex_file)[0]
                relative_path = os.path.join(module_name, file_name_without_ext)
                book_content += f"\\include{{{relative_path}}}\n"

    book_content += "\\end{document}"

    with open(book_path, 'w', encoding='utf-8') as f:
        f.write(book_content)

    logging.info(f"LaTeX book created at {book_path}")

def get_module_structure(base_folder: str) -> Dict[str, List[str]]:
    """Get all PDF files organized by module folders."""
    module_files = {}
    for item in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, item)
        if os.path.isdir(folder_path):
            module_name = item
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            if pdf_files:
                module_files[module_name] = [os.path.join(folder_path, pdf) for pdf in pdf_files]
    return module_files

def main():
    input_folder = "/Users/dietmar/Dropbox/pdfs2to_be_evaluated"
    output_folder = "/Users/dietmar/Dropbox/pdf_reviews"
    book_path = "/Users/dietmar/Dropbox/pdf_reviews/book.tex"

    module_files = get_module_structure(input_folder)
    batch_review_slides(module_files, output_folder)
    create_latex_book(output_folder, book_path)
    print(f"\nBatch review completed. Reviews saved in {output_folder}")
    print(f"LaTeX book created at {book_path}")

if __name__ == "__main__":
    main()
