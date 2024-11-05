# Investment-Evaluator
Evaluate whether you should invest in a company just by uploading a pdf

# Financial Document Analysis using GPT

## Overview

This project implements an algorithm to extract and summarize key information from financial PDFs, particularly earnings call transcripts, to aid investors in evaluating companies. The main objectives are to identify future growth prospects, key changes in business strategies, triggers for significant changes, and important information that could affect future earnings and growth. 

The solution leverages the Pegasus model for summarization and utilizes the GPT API to analyze the extracted content for actionable insights.

## Features

- Extracts text from PDF documents.
- Summarizes text using NLP models.
- Provides structured financial analysis with actionable insights.
- Works with any financial document of similar structure.

## Requirements

- Python 3.x
- Required packages:
  - `transformers`
  - `nltk`
  - `torch`
  - `PyPDF2`
  - `requests`
  
## Installation

To install the required packages, you can run:

```bash
pip install transformers nltk torch PyPDF2 requests
```

## Code Explanation

### 1. Importing Libraries

```python
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize
import re
import time
from typing import List, Dict, Any
import PyPDF2
import requests
from google.colab import files
import json
```

Here, we import necessary libraries for text processing, PDF extraction, natural language processing, and HTTP requests. 

### 2. DialogueSummarizer Class

```python
class DialogueSummarizer:
    def __init__(self, chunk_size: int = 3000):
        # Initialization and model loading
```

The `DialogueSummarizer` class initializes the Pegasus model for text summarization and defines parameters for chunk size.

#### 2.1 Text Extraction from PDF

```python
def extract_text_from_pdf(self, pdf_path: str) -> str:
    # Extracts text from the PDF file
```

This method reads the PDF and extracts text while preserving the structure.

#### 2.2 Parsing Dialogue

```python
def parse_dialogue(self, text: str) -> List[Dict]:
    # Parses the text into structured dialogue segments
```

The `parse_dialogue` method segments the extracted text into speaker-content pairs.

#### 2.3 Text Cleaning

```python
def clean_text(self, text: str) -> str:
    # Cleans and preprocesses the text
```

This method removes unnecessary whitespace and formats the text for analysis.

#### 2.4 Chunking Segments

```python
def split_into_chunks(self, dialogue_segments: List[Dict]) -> List[List[Dict]]:
    # Splits dialogue into manageable chunks
```

It divides the dialogue segments into chunks to avoid exceeding model input limits.

#### 2.5 Summarization

```python
def summarize_chunk(self, chunk: List[Dict], max_length: int = 150) -> str:
    # Summarizes a chunk of dialogue
```

This method uses the Pegasus model to generate a summary of each chunk.

### 3. GPTAnalyzer Class

```python
class GPTAnalyzer:
    def __init__(self, api_key: str):
        # Initializes GPT analyzer with API key
```

The `GPTAnalyzer` class handles the interaction with the GPT API for deeper analysis of the summarized text.

#### 3.1 Creating Analysis Prompt

```python
def create_analysis_prompt(self, text: str) -> str:
    # Creates a detailed prompt for GPT analysis
```

This method formulates a structured prompt that guides the GPT model to provide focused insights based on the financial document.

#### 3.2 Retrieving GPT Analysis

```python
def get_gpt_analysis(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
    # Retrieves the analysis from GPT API
```

This method interacts with the GPT API and handles potential errors with a retry mechanism.

### 4. Main Function

```python
def main():
    # The entry point for processing the PDF document
```

The `main` function ties together the summarization and analysis processes. It prompts the user to upload a PDF file, processes it, and outputs both the summarized and analyzed results.

### 5. Execution

```python
if __name__ == "__main__":
    # Run the main function
```

This block checks if the script is being run directly and executes the main function.

## Usage

1. Run the script.
2. Upload a PDF file containing financial data.
3. Review the generated summaries and investment analysis.

## Conclusion

This project demonstrates the application of advanced NLP techniques to enhance the accessibility and interpretability of complex financial documents, providing valuable insights for investors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
