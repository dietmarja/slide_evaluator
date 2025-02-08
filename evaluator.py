from openai import OpenAI
import os
import re
import logging
import base64
from pdf2image import convert_from_path
import PyPDF2
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='slide_review.log')

# OpenAI API configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Review prompt template without LaTeX formatting
REVIEW_PROMPT = """
Consider the learning materials represented as a pdf file attached and 
put together a critical evaluation along the lines of the criteria below:

Consistency and Alignment
- Do the slides align with the Learning Outcomes and module handbook description in terms of content, depth, and focus?
- Do the assessments (labs, CA, and exams) reflect the skills and knowledge outlined in the module learning outcomes?
- Are key concepts and skills introduced progressively across the module in a way that supports student learning?

Assessment and Lab Effectiveness
- Do the labs effectively prepare students for their project work and support their understanding of key concepts?
- Do the exams and CA build on each other to provide a structured approach to learning and assessment?
- Are the types of questions in the exams reflective of the skills students need for the project and continuous assessments?

Clarity and Understanding
- Are the slides structured logically, ensuring that concepts build on each other in a way that supports understanding?
- Is there unnecessary repetition in the slides that could be streamlined?
- Are the explanations clear and appropriate for the students' level, avoiding excessive complexity or oversimplification?
- Are key concepts illustrated effectively through examples, diagrams, or case studies?
- Are important terms, formulas, and concepts adequately explained with definitions and context?

Accuracy and Completeness
- Are the slides factually accurate, avoiding outdated or incorrect information?
- Are there any gaps in the slides where additional explanation or context would improve understanding?
- Are important distinctions and nuances in the subject matter properly addressed?

Engagement and Effectiveness of Delivery
- Do the slides incorporate elements that make them engaging and interactive (e.g., thought-provoking questions, activities, real-world applications)?
- Are technical concepts explained in a way that encourages engagement and critical thinking rather than passive memorization?
- Is there sufficient scaffolding for difficult topics, ensuring that students have the necessary background before encountering advanced concepts?

Synchronous and Asynchronous Learning
- Are the slides suitable for synchronous learning?
- Do the slides encourage and include offerings for asynchronous learning?

Critique and Improvements
- Based on the analysis of the slides, what improvements can be made to enhance clarity, engagement, and effectiveness?
- Are there better ways to organize or structure the content to support student learning?
- Are there any missing topics or underexplored areas that should be covered in greater depth?

Avoid general statements like "Could profit from better scaffolding". Instead, offer specific suggestions on what can actually be done to improve the slides on the point flagged up.

In the summary, highlight the 3 areas that most urgently need attention.

Slide Content:
{slide_text}
"""

def clean_latex_content(content):
    """
    Clean LaTeX content to ensure proper list environments and restrict commands.
    Prevents nested itemize environments and ensures proper closure.
    """
    # First handle any markdown headers
    def convert_markdown_headers(match):
        hashes = match.group(1)
        title = match.group(2).strip()
        if len(hashes) == 3:
            return f"\\subsection{{{title}}}"
        elif len(hashes) in [1, 2]:
            return f"\\section{{{title}}}"
        return title

    # Convert markdown headers
    content = re.sub(r'^(#{1,3})\s*(.+?)$', convert_markdown_headers, content, flags=re.MULTILINE)
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)

    # Remove nested itemize environments
    def flatten_itemize(text):
        # Remove multiple begins
        while '\\begin{itemize}\\begin{itemize}' in text:
            text = text.replace('\\begin{itemize}\\begin{itemize}', '\\begin{itemize}')

        # Remove multiple ends
        while '\\end{itemize}\\end{itemize}' in text:
            text = text.replace('\\end{itemize}\\end{itemize}', '\\end{itemize}')

        return text

    # Process the content line by line
    lines = content.split('\n')
    processed_lines = []
    in_itemize = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Handle begin itemize
        if '\\begin{itemize}' in line:
            if not in_itemize:
                processed_lines.append('\\begin{itemize}')
                in_itemize = True
            continue

        # Handle end itemize
        if '\\end{itemize}' in line:
            if in_itemize:
                processed_lines.append('\\end{itemize}')
                in_itemize = False
            continue

        # Handle items
        if '\\item' in line:
            if not in_itemize:
                processed_lines.append('\\begin{itemize}')
                in_itemize = True
            processed_lines.append(line)
            continue

        # Handle other allowed LaTeX commands
        if line.startswith('\\section{') or line.startswith('\\subsection{') or line.startswith('\\chapter{'):
            if in_itemize:
                processed_lines.append('\\end{itemize}')
                in_itemize = False
            processed_lines.append(line)
            continue

        # Handle regular text
        if in_itemize and not line.startswith('\\'):
            # Append text to previous item if it exists
            if processed_lines and '\\item' in processed_lines[-1]:
                processed_lines[-1] = processed_lines[-1] + ' ' + line
            continue
        elif not in_itemize and not line.startswith('\\'):
            processed_lines.append(line)

    # Ensure we close any open itemize environment
    if in_itemize:
        processed_lines.append('\\end{itemize}')

    # Join lines and clean up
    content = '\n'.join(processed_lines)

    # Remove any remaining nested environments
    content = flatten_itemize(content)

    # Remove empty itemize environments
    content = re.sub(r'\\begin{itemize}\s*\\end{itemize}', '', content)

    # Clean up multiple newlines
    content = re.sub(r'\n\s*\n', '\n\n', content)

    return content.strip()

def generate_slide_review(slide_text):
    """Generate a review for the slides using OpenAI API."""
    try:
        system_prompt = """You are an expert educational content reviewer analyzing PowerPoint slides.
Generate your review using LaTeX sections and subsections (not markdown headers).
Use \section{} for main sections and \subsection{} for subsections.
For lists, use \begin{itemize} \item ... \end{itemize}.
Do not use any other LaTeX commands."""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": REVIEW_PROMPT.format(slide_text=slide_text)}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        return clean_latex_content(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error generating review: {e}")
        return "Error generating review. Please check the input PDF."

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def pdf_to_images(pdf_path, max_pages=10):
    """Convert PDF pages to images."""
    try:
        # Create a temporary directory for images
        os.makedirs('temp_pdf_images', exist_ok=True)

        # Convert PDF to images
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)

        image_paths = []
        for i, image in enumerate(images):
            image_path = f'temp_pdf_images/page_{i+1}.png'
            image.save(image_path, 'PNG')
            image_paths.append(image_path)

        return image_paths
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return []

def encode_image(image_path):
    """Encode an image to base64 for GPT-4 Vision API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_pdf_visuals(pdf_path):
    """Analyze visual aspects of PDF slides using GPT-4 Vision."""
    try:
        # Convert PDF to images
        image_paths = pdf_to_images(pdf_path)

        if not image_paths:
            return ""

        # Analyze first few pages (limit to 5 for API cost management)
        visual_analysis_prompts = [
            "Analyze the overall visual design and layout of these slides. Consider color scheme, font choices, consistency, and visual hierarchy.",
            "Examine the use of graphics, charts, and diagrams. Are they clear, informative, and effectively integrated?",
            "Assess the typography and text presentation. Is the text readable? Are font sizes and styles appropriate?",
            "Evaluate the use of white space, alignment, and overall visual balance of the slides.",
            "Identify any visual design elements that enhance or detract from the slides' educational effectiveness."
        ]

        full_visual_analysis = ""

        for i, (image_path, prompt) in enumerate(zip(image_paths[:5], visual_analysis_prompts), 1):
            base64_image = encode_image(image_path)

            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=300
            )

            page_analysis = f"\\subsection{{Page {i} Analysis}}\n{response.choices[0].message.content}\n\n"
            full_visual_analysis += clean_latex_content(page_analysis)

        return full_visual_analysis

    except Exception as e:
        logging.error(f"Error analyzing PDF visuals: {e}")
        return ""


def get_module_structure(base_folder):
    """Get all PDF files organized by module folders."""
    module_files = {}
    for item in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, item)
        if os.path.isdir(folder_path):
            module_name = item
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            if pdf_files:  # Only include folders that contain PDFs
                module_files[module_name] = [os.path.join(folder_path, pdf) for pdf in pdf_files]
    return module_files

def batch_review_slides(module_files, output_folder):
    """Batch process PDF slides in module folders and save reviews."""
    os.makedirs(output_folder, exist_ok=True)
    
    for module_name, pdf_files in module_files.items():
        # Create module subfolder in output
        module_output = os.path.join(output_folder, module_name)
        os.makedirs(module_output, exist_ok=True)
        
        for pdf_path in pdf_files:
            try:
                filename = os.path.basename(pdf_path)
                slide_text = extract_pdf_text(pdf_path)
                review = generate_slide_review(slide_text)
                visual_analysis = analyze_pdf_visuals(pdf_path)
                combined_review = f"{review}\n\n\\section{{Visual Analysis}}\n{visual_analysis}"
                combined_review = clean_latex_content(combined_review)
                
                chapter_title = os.path.splitext(filename)[0]
                output_filename = os.path.splitext(filename)[0] + '.tex'
                output_path = os.path.join(module_output, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"\\chapter{{{chapter_title}}}\n")
                    f.write(combined_review.replace("&", "\\&"))
                
                logging.info(f"Processed {filename} in module {module_name}")
                
            except Exception as e:
                logging.error(f"Error processing {filename} in module {module_name}: {e}")

def create_latex_book(output_folder, book_path):
    """Create a LaTeX book document that imports all generated .tex files."""
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start the LaTeX document with additional packages
    book_content = """\\documentclass{book}
\\usepackage[margin=1in]{geometry}
\\usepackage{datetime}
\\usepackage[colorlinks=true, 
            linkcolor=blue, 
            urlcolor=blue,
            bookmarks=true,
            bookmarksopen=true]{hyperref}
\\title{Digital4Business -- Slide Evaluations}
\\author{}
\\date{\\vspace{1em}Generated on: %s}
\\begin{document}
\\maketitle

\\chapter*{Preface}
This book offers evaluations of the course material of Digital4Business Master's Course. 
All evaluations have been done by ChatGPT following criteria that can be found in Chapter 1.

\\tableofcontents

\\chapter{Evaluation Criteria}
\\begin{verbatim}
%s
\\end{verbatim}
""" % (timestamp, REVIEW_PROMPT.replace("Slide Content:\n{slide_text}", "").replace("&", "\\&"))

    # Process each module folder
    for module_name in sorted(os.listdir(output_folder)):
        module_path = os.path.join(output_folder, module_name)
        if os.path.isdir(module_path):
            # Add part for module
            book_content += f"\n\\part{{Module {module_name}}}\n"
            
            # Process tex files in the module folder
            tex_files = [f for f in os.listdir(module_path) if f.endswith('.tex')]
            for tex_file in sorted(tex_files):
                file_name_without_ext = os.path.splitext(tex_file)[0]
                # Use relative path from book.tex location to module folder
                relative_path = os.path.join(module_name, file_name_without_ext)
                book_content += f"\\include{{{relative_path}}}\n"
    
    book_content += "\\end{document}"
    
    with open(book_path, 'w', encoding='utf-8') as f:
        f.write(book_content)
    
    logging.info(f"LaTeX book created at {book_path}")

if __name__ == "__main__":
    input_folder = "/Users/dietmar/Dropbox/pdfs2to_be_evaluated"
    output_folder = "/Users/dietmar/Dropbox/pdf_reviews"
    book_path = "/Users/dietmar/Dropbox/pdf_reviews/book.tex"

    # Get module structure
    module_files = get_module_structure(input_folder)
    
    # Process all slides
    batch_review_slides(module_files, output_folder)
    
    # Create the book
    create_latex_book(output_folder, book_path)
    print(f"Batch review completed. Reviews saved in {output_folder}")
    print(f"LaTeX book created at {book_path}")