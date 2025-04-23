# PALMS: Pedagogical Assessment via Language Models and Statistical Analysis

**PALMS** is a Python-based tool that automatically evaluates teaching slides currently using OpenAI's GPT-4 and generates comprehensive LaTeX reports. 
It is designed to support educational institutions (VET or HE) in assessing course materials against pedagogical standards and learning objectives.

---

## 📘 Description

PALMS (Pedagogical Assessment via Language Models and Statistical Analysis) analyzes teaching materials such as PDF slide decks and produces detailed LaTeX reports evaluating:

- **Content alignment** with learning outcomes
- **Visual design** and layout effectiveness
- **Assessment and lab** integration
- **Pedagogical clarity**, completeness, and accuracy
- **Student engagement** features and scaffolding
- **Support for synchronous and asynchronous learning**

PALMS can optionally assess content against known standards like the **EQF** or domain-specific benchmarks, either using its pretrained knowledge or with custom criteria provided by the user.

---

## ✨ Features

- AI-driven content evaluation using **GPT-4**
- Batch processing of multiple PDF slide decks
- Structured LaTeX reports with clickable Table of Contents
- Module-level organization for large courses
- Visual design analysis (basic image support)
- Comprehensive **scoring system** per slide and slide deck
- Automated generation of **bar and line charts** for score visualization
- Customizable evaluation criteria

---

## 🛠 Requirements

- Python 3.8+
- OpenAI API key
- TeX Live or similar LaTeX distribution
- Poppler (for `pdf2image`)
- Python packages (install via `requirements.txt`):
  ```bash
  openai>=1.0.0
  PyPDF2>=3.0.0
  pdf2image>=1.16.0
