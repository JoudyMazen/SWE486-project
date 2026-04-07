# Sentiment Analysis: Local vs Cloud Deployment
**SWE 486 - Cloud Computing & Big Data | King Saud University**

## Project Overview
This project compares the performance of sentiment analysis using 
local deployment (DistilBERT on device) versus cloud deployment 
(AWS Comprehend). We measure inference time, CPU usage, memory 
usage, and energy consumption across short, medium, and large text inputs.

## AI Service
- **Task:** Sentiment Analysis (NLP)
- **Model:** DistilBERT fine-tuned on SST-2
- **Source:** Hugging Face — `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

## Local Deployment
- **Device:** MacBook Air, Apple M4, 16GB RAM
- **Language:** Python 3.14.2
- **Libraries:** Transformers, PyTorch, psutil

## Repository Structure
├── sentiment.py         # Local deployment script
├── short_inputs.txt     # Short input samples (< 50 words)
├── medium_inputs.txt    # Medium input samples (51–150 words)
├── large_inputs.txt     # Large input samples (> 150 words)
└── README.md

## How to Run
1. Create and activate a virtual environment:
   python3 -m venv myenv
   source myenv/bin/activate

2. Install dependencies:
   pip install transformers torch psutil

3. Run the script:
   python sentiment.py

## Group Members
| Name | ID |
|---|---|
| Joudy Alnounou | 444200139 |
| Showq Alhadlaq | 444204111 |
| Sadeem Alsayari | 444201182 |
| Hayfa Alruwaita | 444200468 |
| Albatool Alalsheikh | 435202169 |
| Tala Alrajeh | 444200459 |

**Supervisor:** Ms. Ghadah Alamer  
**Section:** 54978 | **Group:** 2
