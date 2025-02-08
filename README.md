# Slide Evaluator

A Python-based tool that automatically evaluates teaching slides using OpenAI's GPT-4 and generates comprehensive LaTeX reports.

## Description
The Slide Evaluator analyzes teaching materials for higher education courses by:
- Processing PDF slide decks
- Evaluating content quality and structure in line with changeable criteria
- Assessing visual design and layout (in progress)
- Generating detailed LaTeX reports organized by course modules

The system evaluates slides based on multiple criteria including:
- Consistency and alignment with learning outcomes
- Assessment and lab effectiveness
- Clarity and understanding
- Accuracy and completeness
- Engagement and effectiveness of delivery
- Synchronous and asynchronous learning capabilities

## Requirements
- Python 3.8+
- OpenAI API key
- TeXLive or similar LaTeX distribution
- Python packages:
  ```
  openai>=1.0.0
  PyPDF2>=3.0.0
  pdf2image>=1.16.0
  ```

## Installation
1. Clone the repository
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Example Directory Structure
```
pdfs2to_be_evaluated/
├── Blockchain/
│   ├── Week1.pdf
│   └── Week2.pdf
└── Quantum Computing/
    ├── Week1.pdf
    └── Week2.pdf
```

## Usage
1. Place your PDF slides in appropriate module folders under `pdfs2to_be_evaluated/`
2. Run the evaluator:
   ```bash
   python evaluator.py
   ```
3. Find generated reports in `pdf_reviews/`:
   - Individual .tex files (chapters) for each slide deck
   - Complete book.tex combining all chapters 
   - Final PDF with clickable table of contents

## Output Format
The generated report includes:
- Title page with timestamp
- Preface (editable)
- Clickable table of contents
- Evaluation criteria chapter
- Module-organized content
- Visual analysis for each slide deck

## Configuration
Modify paths in evaluator.py:
```python
input_folder = "/Users/yourname/path/to/pdfs2to_be_evaluated"
output_folder = "/Users/yourname/path/to/pdf_reviews"
book_path = "/Users/yourname/path/to/pdf_reviews/book.tex"
```

## Requirements & Limitations
- Currently only processes PDF files
- Requires active internet connection for GPT-4 API
- Visual analysis requires good quality PDFs

## Contributing
Contributions welcome! Please submit pull requests for any improvements.

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
