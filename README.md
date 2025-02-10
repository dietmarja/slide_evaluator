# Course Material Evaluator

An automated system for evaluating teaching materials using GPT-4, with a focus on both content and visual aspects of educational slides.

## Overview

The Course Material Evaluator automatically analyzes teaching materials from PDF slides, generating comprehensive LaTeX reports that assess:
- Content alignment with learning objectives
- Visual design effectiveness for learning
- Assessment and lab effectiveness
- Content clarity and completeness
- Pedagogical effectiveness

## Features

- Automated content analysis using GPT-4
- Visual element analysis of slides
- Pedagogical effectiveness assessment
- Module-based organization
- LaTeX report generation with clickable table of contents
- Batch processing of multiple PDF files

## Requirements

- Python 3.8+
- OpenAI API key
- TeX Live or similar LaTeX distribution
- Poppler (for PDF processing)
- Required Python packages (see `requirements.txt`)

### Code
- evaluator.py  
  Central script for evaluating pdf and creating a LaTeX report

- debug_pdf.py
  PDF Conversion Debug Tool

- test_visual.py
  Visual Analysis Test Suite 



## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/course-evaluator.git
cd course-evaluator
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

4. Set up OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

1. Organize PDF slides in module folders:
```
pdfs2to_be_evaluated/
├── Module1/
│   ├── slides1.pdf
│   └── slides2.pdf
└── Module2/
    ├── slides1.pdf
    └── slides2.pdf
```

2. Run the evaluator:
```bash
python evaluator.py
```

3. Find generated reports in output directory:
- Individual .tex files for each slide deck
- Complete book.tex combining all evaluations
- Final PDF with table of contents

## Configuration

Edit paths in `evaluator.py`:
```python
input_folder = "/path/to/pdfs2to_be_evaluated"
output_folder = "/path/to/pdf_reviews"
book_path = "/path/to/pdf_reviews/book.tex"
```

## Output Structure

The generated report includes:
- Title page with timestamp
- Preface explaining methodology
- Evaluation criteria
- Module-based content organization
- Content and visual analysis for each slide deck
- Specific recommendations for improvements

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
