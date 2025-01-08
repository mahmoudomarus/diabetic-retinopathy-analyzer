# Diabetic Retinopathy Analyzer

An AI-powered system for analyzing retinal images and detecting diabetic retinopathy with integrated LLM analysis.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Move your model file:
```bash
# Move pretrained model to models directory
mv enhanced_retinal_analyzer_epoch_10.pth models/
```

3. Run the app:
```bash
streamlit run app.py
```

## Features

- Advanced retinal image analysis using ViT and EfficientNet
- Detailed medical report generation
- LLM-powered analysis using GPT-4
- Vessel metrics analysis
- Report saving and export

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional)
- OpenAI API key for LLM analysis