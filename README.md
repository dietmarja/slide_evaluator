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
- Comprehensive scoring system for individual slide and slide decks 
  (lectures) averaged across all evalation criteria
- Generation of bar and line charts for visualizing scores

## Requirements

- Python 3.8+
- OpenAI API key
- TeX Live or similar LaTeX distribution
- Poppler (for PDF processing)
- Required Python packages (see `requirements.txt`)

### Code
- **evaluator.py**
  Central script for evaluating PDFs and creating a LaTeX report

- **debug_pdf.py**
  PDF Conversion Debug Tool

- **test_visual.py**
  Visual Analysis Test Suite

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/course-evaluator.git
cd course-evaluator
