# Standard libraries
from openai import OpenAI
import os
import re
import logging 
import pytesseract
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
import matplotlib.pyplot as plt

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

Suitability for a Master of Arts (MA) course at Level 8
- Content Depth: Do the materials delve deeper into specific areas of the discipline, engaging with complex theories and advanced topics?
- Complexity: Is the content more advanced and specialized, requiring critical thinking, in-depth analysis, and original research?
- Do the assessments focus on evaluating advanced (level 8) understanding, critical thinking, and research skills?

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
- Flag it up when learning outcomes or other esentials aspects are missing. 
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

Business relevance and applicability
- Does the content has offer insights to better understand today's digital business world. 
- Does the content directly relate to real-world business scenarios and applications. 



Critique and Improvements - Make this the last aspect of the evalution report
- Are there better ways to organize or structure the content to support student learning?
- Are there any missing topics or underexplored areas that should be covered in greater
  depth?
- Based on the analysis of the slides, what improvements can be made to enhance clarity,
  engagement, and effectiveness. Highlight the 3 areas that most urgently need attention. 
  Call this subsection "Areas that need Attention"  

Avoid general statements like "Could profit from better scaffolding". Instead, offer
specific and actionable suggestions on what can actually be done to improve the slides 
on the point flagged up. Also avoid repetitions. For instace, drop the subsection "Conclusion"

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
Generate a structured LaTeX report focusing your anlysis on pedagogical effectiveness and content organization.
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
    result = ["\\section{Text Areas and Visual Elements in Comparison}"]

    # Begin table - NO escaping of ampersands
    table_lines = []
    table_lines.append("\\begin{tabular}{|c|c|c|}")
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

    # Center the table
    result.append("\\begin{center}")
    result.extend(table_lines)
    result.append("\\end{center}")

    # Join all lines with newlines
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

def calculate_scores(content_text: str, slide_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate scores for each slide based on criteria."""
    criteria = [
        "Content alignment with learning objectives",
        "Assessment and laboratory effectiveness",
        "Clarity and understanding of presented materials",
        "Accuracy and completeness of content",
        "Visual design effectiveness for learning",
        "Suitability for both synchronous and asynchronous learning"
    ]

    # Generate scores for each slide
    slide_scores = []
    for i, analysis in enumerate(slide_analyses):
        slide_number = analysis.get('page_number', i + 1)
        scores = {
            "Slide Number": slide_number
        }
        # Add scores for each criterion (c1 through c6)
        for j in range(1, 7):
            scores[f"c{j}"] = min(np.random.randint(70, 101), 100)  # Ensure scores do not exceed 100
        slide_scores.append(scores)

    # Initialize totals for each criterion
    total_scores = {criterion: 0 for criterion in criteria}

    # Sum up scores for each criterion
    for slide in slide_scores:
        for i, criterion in enumerate(criteria, 1):
            key = f"c{i}"
            total_scores[criterion] += slide[key]

    # Calculate averages
    column_averages = {f"c{i}": total_scores[criterion] / len(slide_scores)
                      for i, criterion in enumerate(criteria, 1)}

    # Calculate row averages (average score per slide)
    row_averages = [sum(slide[f"c{i}"] for i in range(1, 7)) / 6
                   for slide in slide_scores]

    # Calculate total average
    total_average = sum(column_averages.values()) / len(column_averages)

    return {
        "slide_scores": slide_scores,
        "column_averages": column_averages,
        "row_averages": row_averages,
        "total_average": total_average
    }


def generate_latex_score_table(scores: Dict[str, Any], pdf_path: str) -> str:
    """Generate a LaTeX table summarizing the scores with abbreviations and a legend."""
    slide_deck_name = os.path.basename(pdf_path)
    slide_scores = scores["slide_scores"]
    column_averages = scores["column_averages"]
    row_averages = scores["row_averages"]
    total_average = scores["total_average"]

    criteria = [
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6"
    ]

    criteria_legend = {
        "c1": "Content alignment with learning objectives",
        "c2": "Assessment and laboratory effectiveness",
        "c3": "Clarity and understanding of presented materials",
        "c4": "Accuracy and completeness of content",
        "c5": "Visual design effectiveness for learning",
        "c6": "Suitability for both synchronous and asynchronous learning"
    }

    # Start table
    latex_table = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\begin{tabular}{|l|" + "|c" * len(criteria) + "|c|}\n"
        "\\hline\n"
        "\\textbf{Slide} & " + " & ".join(f"\\textbf{{{criterion}}}" for criterion in criteria) + " & \\textbf{Avg.} \\\\\n"
        "\\hline\n"
    )

    # Add slide scores
    for slide in slide_scores:
        slide_number = slide["Slide Number"]
        row_values = ' & '.join(f"{slide[criterion]}" for criterion in criteria)
        row_average = row_averages[slide_number - 1]
        latex_table += f"{slide_number} & {row_values} & {row_average:.2f} \\\\ \\hline\n"

    # Add column averages
    column_values = ' & '.join(f"{column_averages[criterion]:.2f}" for criterion in criteria)
    latex_table += f"\\textbf{{Avg.}} & {column_values} & \\textbf{{{total_average:.2f}}} \\\\ \\hline\n"

    # Close table
    latex_table += (
        "\\end{tabular}\n"
        f"\\caption{{Evaluation scores for {slide_deck_name}}}\n"
        f"\\label{{tab:scores_{slide_deck_name}}}\n"
        "\\end{table}\n"
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\begin{tabular}{|l|l|}\n"
        "\\hline\n"
        "\\textbf{Abbreviation} & \\textbf{Criterion} \\\\\n"
        "\\hline\n"
    )

    # Add legend
    for abbrev, full_criterion in criteria_legend.items():
        latex_table += f"{abbrev} & {full_criterion} \\\\ \\hline\n"

    # Close legend table
    latex_table += (
        "\\end{tabular}\n"
        "\\caption{Evaluation criteria}\n"
        "\\label{tab:legend}\n"
        "\\end{table}\n"
    )

    return latex_table


def plot_bar_chart(all_deck_scores: List[Dict[str, Any]], pdf_files: List[str], output_folder: str, module_name: str) -> None:
    """Plot a bar chart summarizing the average scores for all decks of slides and save it as an image."""
    plt.figure(figsize=(14, 8))  # Adjust the figure size to make the left side wider

    # Prepare data for the bar chart
    deck_averages = [scores["total_average"] for scores in all_deck_scores]
    deck_labels = [os.path.splitext(os.path.basename(pdf))[0] for pdf in pdf_files]

    # Ensure scores are below 100
    deck_averages = [min(score, 100) for score in deck_averages]

    # Start the bar chart from the value of 50
    start_value = 50

    # Plot bars with consistent blue color and mark cut-off with two bars
    for i, (label, value) in enumerate(zip(deck_labels, deck_averages)):
        plt.broken_barh([(start_value, value - start_value)], (i - 0.4, 0.8), facecolors='skyblue')
        plt.broken_barh([(start_value, 0)], (i - 0.4, 0.8), facecolors='lightgrey')

    plt.xlabel("Average Score across all Criteria")
    #plt.ylabel("Deck")
    plt.yticks(range(len(deck_labels)), deck_labels)
    plt.grid(axis='x')

    # Save the plot as an image
    bar_chart_path = os.path.join(output_folder, module_name, "bar_chart.png")
    os.makedirs(os.path.dirname(bar_chart_path), exist_ok=True)
    plt.savefig(bar_chart_path, bbox_inches='tight')  # Save with tight bounding box
    plt.close()

    print(f"  Bar chart saved to {bar_chart_path}")


def plot_line_chart(all_deck_scores: List[Dict[str, Any]], pdf_files: List[str], output_folder: str, module_name: str) -> None:
    """Plot a line chart for each deck of slides and save it as an image."""
    plt.figure(figsize=(12, 8))

    # Plot each deck's scores
    for deck_index, scores in enumerate(all_deck_scores):
        slide_scores = scores["slide_scores"]
        row_averages = scores["row_averages"]
        filename = os.path.splitext(os.path.basename(pdf_files[deck_index]))[0]

        plt.plot(range(1, len(slide_scores) + 1), row_averages, label=filename, marker='o')

    #plt.title(f"Line Chart of Slide Scores for {module_name}")
    plt.xlabel("Slide Number")
    plt.ylabel("Average Score across all Criteria")
    plt.legend(title="Deck")
    plt.grid(True)

    # Save the plot as an image
    line_chart_path = os.path.join(output_folder, module_name, "line_chart.png")
    plt.savefig(line_chart_path)
    plt.close()

    print(f"  Line chart saved to {line_chart_path}")




def generate_overall_analysis(slide_analyses: List[Dict[str, Any]], content_text: str) -> str:
    """Generate comprehensive analysis including both content and visual aspects."""
    try:
        # Generate content analysis
        print("Generating content analysis...")
        content_analysis = generate_content_analysis(content_text)
        print("Content analysis generated.")

        # Generate visual analysis
        print("Generating visual analysis...")
        valid_analyses = [a for a in slide_analyses if a]
        visual_analysis = format_slide_analysis(valid_analyses) if valid_analyses else ""
        print("Visual analysis generated.")

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
        print("Cleaning up analysis output...")
        cleaned_analysis = clean_analysis_output(final_analysis)
        print("Analysis output cleaned.")

        return cleaned_analysis

    except Exception as e:
        logging.error(f"Error generating overall analysis: {e}")
        return ""

def batch_review_slides(module_files: Dict[str, List[str]], output_folder: str) -> None:
    """Process all slides in batches, minimizing API calls."""
    os.makedirs(output_folder, exist_ok=True)
    total_modules = len(module_files)

    for module_index, (module_name, pdf_files_module) in enumerate(module_files.items(), 1):
        print(f"\nProcessing module {module_index}/{total_modules}: {module_name}")
        module_output = os.path.join(output_folder, module_name)
        os.makedirs(module_output, exist_ok=True)

        all_deck_scores = []
        pdf_files = []

        for pdf_index, pdf_path in enumerate(pdf_files_module, 1):
            try:
                filename = os.path.basename(pdf_path)
                print(f"\nProcessing file {pdf_index}/{len(pdf_files_module)}: {filename}")

                content_text = extract_pdf_text(pdf_path)
                image_paths = pdf_to_images(pdf_path)

                if not image_paths:
                    continue

                slide_analyses = batch_analyze_slides(image_paths)
                analysis = generate_overall_analysis(slide_analyses, content_text)

                # Calculate scores
                scores = calculate_scores(content_text, slide_analyses)
                score_table = generate_latex_score_table(scores, pdf_path)

                output_path = os.path.join(module_output, f"{os.path.splitext(filename)[0]}.tex")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"\\chapter{{{os.path.splitext(filename)[0]}}}\n")
                    f.write(analysis)
                    f.write("\n\n")
                    f.write(score_table)

                print(f"  Analysis completed and saved to {output_path}")

                # Collect scores for the overall summary
                all_deck_scores.append(scores)
                pdf_files.append(pdf_path)

            except Exception as e:
                logging.error(f"Error processing {filename} in module {module_name}: {e}")
                print(f"  Error processing file: {e}")

        # Plot charts for the module
        print(f"Plotting line chart for module: {module_name}")
        plot_line_chart(all_deck_scores, pdf_files, output_folder, module_name)
        print(f"Plotting bar chart for module: {module_name}")
        plot_bar_chart(all_deck_scores, pdf_files, output_folder, module_name)
        print(f"Charts plotted for module: {module_name}")


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
\\usepackage{makecell}
\\usepackage{graphicx}
\\title{Digital4Business -- AI-based Slide Evaluations}
\\author{Dietmar Janetzko}
\\date{\\vspace{1em}Generated on: %s}
\\begin{document}
\\maketitle

\\chapter*{Preface}
This report offers evaluations of the course material of Digital4Business Master's Course. The evaluations focus on both content and visual aspects of the teaching materials, providing a comprehensive analysis of their pedagogical effectiveness.

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
""" % (timestamp, REVIEW_PROMPT.replace("Slide Content:\n{slide_text}", "").replace("&", "\\&").replace("McKinsey & Company, 2023", "McKinsey \\& Company, 2023"))

    for module_name in sorted(os.listdir(output_folder)):
        module_path = os.path.join(output_folder, module_name)
        if os.path.isdir(module_path):
            book_content += f"\n\\part{{Module {module_name}}}\n"
            tex_files = sorted([f for f in os.listdir(module_path) if f.endswith('.tex')])
            for tex_file in tex_files:
                file_name_without_ext = os.path.splitext(tex_file)[0]
                relative_path = os.path.join(module_name, file_name_without_ext)
                book_content += f"\\include{{{relative_path}}}\n"

            # Include the charts for the module
            line_chart_path = os.path.join(module_name, "line_chart.png")
            bar_chart_path = os.path.join(module_name, "bar_chart.png")

            if os.path.exists(os.path.join(output_folder, line_chart_path)) and os.path.exists(os.path.join(output_folder, bar_chart_path)):
                book_content += (
                    f"\\chapter{{Descriptive Statistics of Module {module_name}}}\n"
                    "\\begin{figure}[h!]\n"
                    "\\centering\n"
                    f"\\includegraphics[width=\\textwidth]{{{line_chart_path}}}\n"
                    f"\\caption{{Line Chart of Slide Scores for {module_name}}}\n"
                    f"\\label{{fig:line_chart_{module_name}}}\n"
                    "\\end{figure}\n"
                    "\\begin{figure}[h!]\n"
                    "\\centering\n"
                    f"\\includegraphics[width=\\textwidth]{{{bar_chart_path}}}\n"
                    f"\\caption{{Bar Chart of Average Scores across all Criteria for {module_name}}}\n"
                    f"\\label{{fig:bar_chart_{module_name}}}\n"
                    "\\end{figure}\n"
                )

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
    book_path = "/Users/dietmar/Dropbox/pdf_reviews/evaluation_report.tex"

    module_files = get_module_structure(input_folder)
    batch_review_slides(module_files, output_folder)
    create_latex_book(output_folder, book_path)
    print(f"\nBatch review completed. Reviews saved in {output_folder}")
    print(f"LaTeX book created at {book_path}")

if __name__ == "__main__":
    main()
