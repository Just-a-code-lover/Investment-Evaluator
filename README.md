# 📊 Financial Dialogue Analysis Tool 💼

## 🚀 Overview
Transform lengthy earnings call transcripts into **actionable investment insights** using the power of advanced **NLP** and **AI models**! Our tool combines the precision of **PEGASUS** summarization with the analytical capabilities of **GPT-4** to deliver comprehensive financial analysis. 🎯

## ✨ Key Features

### 1. 📝 Intelligent Document Processing
* 📄 Seamless PDF transcript extraction
* 👥 Smart speaker-dialogue parsing
* 🧹 Advanced text cleaning and preprocessing

### 2. 🤖 Advanced AI Analysis
* 🎯 PEGASUS financial summarization
* 🧠 GPT-4 powered investment insights
* 📈 Structured analytical output

### 3. 💡 Smart Insights Generation
* 🔍 Growth prospect analysis
* 🔄 Business change identification
* ⚡ Investment catalyst detection
* 📊 Financial metric evaluation

### 4. 🛡️ Robust Processing
* ⚙️ Intelligent chunk management
* 🔄 Context preservation
* 🎯 Error handling & retry mechanisms

## 🏗️ Technical Architecture

### 🧠 Core Models

#### 📚 PEGASUS Financial Summarizer
* 🎯 Fine-tuned on Bloomberg financial articles
* 💼 Specialized in financial dialogue
* 📊 Preserves key metrics and context

#### 🤖 GPT-4 Analyzer
* 📈 Pattern recognition
* 🎯 Trend analysis
* ⚖️ Risk assessment
* 💡 Strategic insights

## 🛠️ Technology Stack

### 📚 Libraries & Frameworks
* 🐍 PyTorch & Transformers
* 📄 PyPDF2
* 🔤 NLTK
* 🌐 Requests

### 🔧 Processing Pipeline
```mermaid
graph LR
    A[PDF Input] --> B[Text Extraction]
    B --> C[Dialogue Parsing]
    C --> D[Chunk Processing]
    D --> E[PEGASUS Summarization]
    E --> F[GPT-4 Analysis]
    F --> G[Investment Insights]
```

## 📊 Analysis Categories

### 1. 📈 Growth Analysis
* 🎯 Market expansion opportunities
* 🆕 Product development initiatives
* 💹 Revenue growth drivers

### 2. 🔄 Business Evolution
* 👥 Management changes
* 🛠️ Operational updates
* 📋 Strategic shifts

### 3. ⚡ Investment Catalysts
* 🎯 Key milestones
* 🤝 Strategic partnerships
* 💡 Market opportunities

### 4. 📊 Financial Metrics
* 💰 Revenue analysis
* 📈 Margin trends
* 💼 Cost management

## 🚀 Getting Started

### 📋 Prerequisites
```bash
pip install transformers nltk torch PyPDF2 requests
```

### 🎯 Quick Start
```python
# Initialize analyzer
analyzer = FinancialAnalyzer(api_key="your_key")

# Process document
results = analyzer.process_document("earnings_call.pdf")

# Get insights
insights = results.get_investment_analysis()
```

## 💡 Use Cases

### 1. 📊 Investment Research
* 🔍 Due diligence
* 📈 Market analysis
* 💼 Competitive assessment

### 2. 📋 Financial Planning
* 🎯 Strategy development
* 💡 Risk assessment
* 📊 Performance tracking

## ✨ Benefits

### 1. ⚡ Efficiency
* 🚀 Rapid processing
* 🎯 Automated insights
* 📊 Structured output

### 2. 🎯 Accuracy
* 🤖 Multi-model validation
* 📈 Context preservation
* 💡 Financial expertise

### 3. 💼 Actionability
* 📋 Clear recommendations
* ⚖️ Risk awareness
* 🎯 Strategic focus

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details

## 🤝 Contributing
We welcome contributions! Please feel free to submit a Pull Request.

## 📧 Contact
For any queries, please reach out to me.

---
⭐ Don't forget to star this repo if you found it useful! ⭐

# 🔍 Code Deep Dive: Financial Dialogue Analysis Tool

## 📚 Imports and Setup
```python
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize
import re, time, json
from typing import List, Dict, Any
import PyPDF2
import requests
from google.colab import files
```

### 🛠️ What's Being Imported?
* 🤖 **torch**: Powers our deep learning magic
* 🔄 **transformers**: Houses our PEGASUS model
* 📝 **nltk**: Natural language toolkit for text processing
* 📄 **PyPDF2**: PDF handling wizard
* 🌐 **requests**: For smooth API communication

## 🎯 DialogueSummarizer Class

### 🚀 Initialization
```python
def __init__(self, chunk_size: int = 3000):
    # NLTK setup and model loading
    self.model_name = "human-centered-summarization/financial-summarization-pegasus"
    self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
    self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
```

#### ⚙️ What's Happening?
* 📊 Sets default chunk size (3000 characters)
* 🤖 Loads our fine-tuned PEGASUS model
* 🔧 Prepares tokenizer for text processing
* 💻 Configures GPU/CPU device settings

## 📄 PDF Processing Pipeline

### 1️⃣ Text Extraction
```python
def extract_text_from_pdf(self, pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text
```

#### 🔍 How It Works:
* 📂 Opens PDF in binary mode
* 📃 Processes each page
* 📝 Extracts and combines text
* ✨ Maintains formatting with newlines

### 2️⃣ Dialogue Parsing
```python
def parse_dialogue(self, text: str) -> List[Dict]:
    dialogue_segments = []
    lines = text.split('\n')
    current_speaker = None
    current_content = []
```

#### 🎯 Key Features:
* 👥 Identifies speakers using regex
* 💬 Maintains dialogue structure
* 📝 Groups related content
* 🔄 Handles multi-line statements

## 🧹 Text Processing

### 1️⃣ Text Cleaning
```python
def clean_text(self, text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'\n', '', text)    # Remove newlines
    return text.strip()
```

#### ✨ Cleaning Operations:
* 🧼 Removes extra whitespace
* 🔄 Normalizes line breaks
* ✂️ Trims leading/trailing spaces
* 📏 Standardizes formatting

### 2️⃣ Chunk Creation
```python
def split_into_chunks(self, dialogue_segments: List[Dict]) -> List[List[Dict]]:
    chunks = []
    current_chunk = []
    current_length = 0
```

#### 📦 Chunking Logic:
* 📏 Respects maximum size limits
* 💬 Preserves dialogue context
* 🔄 Maintains speaker attribution
* ✂️ Smart splitting at natural breaks

## 🤖 PEGASUS Summarization

### 1️⃣ Chunk Processing
```python
def summarize_chunk(self, chunk: List[Dict], max_length: int = 150) -> str:
    formatted_text = self.format_chunk_for_summarization(chunk)
    inputs = self.tokenizer(formatted_text, return_tensors="pt", truncation=True)
```

#### 🎯 Parameters:
* 📏 `max_length`: Summary length cap
* 🔍 `num_beams`: Search beam width
* ⚖️ `length_penalty`: Output length control
* 🎚️ `early_stopping`: Generation control

#### ✨ Features:
* 🎯 Financial-specific summarization
* 💡 Context preservation
* 📊 Key information retention
* 🔄 Coherent output generation

## 🧠 GPT Analysis System

### 1️⃣ Analyzer Setup
```python
class GPTAnalyzer:
    def __init__(self, api_key: str):
        self.url = "https://chatgpt-42.p.rapidapi.com/conversationgpt4-2"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "chatgpt-42.p.rapidapi.com"
        }
```

#### 🔧 Configuration:
* 🔑 API authentication
* 🌐 Endpoint setup
* 📝 Content type specification
* 🔄 Retry mechanism configuration

### 2️⃣ Analysis Generation
```python
def get_gpt_analysis(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
    prompt = self.create_analysis_prompt(text)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
```

#### 🎯 Analysis Focus Areas:
* 📈 Growth prospects
* 🔄 Business changes
* ⚡ Investment catalysts
* 📊 Financial metrics
* ⚖️ Risk assessment

## 🚀 Main Execution Flow

```python
def main():
    # Initialize components
    summarizer = DialogueSummarizer(chunk_size=1000)
    analyzer = GPTAnalyzer(api_key)
```

### 📋 Process Steps:
1. 📂 File upload handling
2. 📝 Text extraction
3. 💬 Dialogue parsing
4. 📊 Chunk processing
5. 🤖 Summarization
6. 🧠 GPT analysis
7. 📈 Results formatting

### ⚡ Error Handling:
* 🔄 Retry mechanisms
* 🛡️ Exception capture
* 📝 Error logging
* 🔧 Graceful fallbacks

## 🎯 Output Format

### 1️⃣ Document Results
* 📄 Dialogue segments
* 📊 Chunk summaries
* 📝 Combined summary

### 2️⃣ Analysis Results
* 📈 Growth insights
* 💼 Business changes
* ⚡ Investment triggers
* 📊 Financial metrics
* 🎯 Action recommendations

---
⭐ Code maintained and documented with ❤️ for clarity and usability ⭐
This code creates a complete pipeline for analyzing financial dialogues, from PDF extraction through summarization to final investment analysis, with robust error handling and clear output formatting.
