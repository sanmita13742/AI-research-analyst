# 🤖 AI Research Analyst Pro

**Advanced Chain-of-Prompts Analysis with Free AI Models**

> **Hi! I made this project for all those innovators who are stuck reading long papers of research articles.**
> 
> AI Research Analyst Pro not only reads the article and provides a summary report with step-by-step methodology, it also compares it with your new project idea and gives out an evaluation report that includes strengths, weaknesses, and potential improvements - and much more which you can download as PDF or TXT!
> 
> It also lets you choose your own model (Ollama, Groq, Gemini) with your own API key (all free tiers).

A  Streamlit application that analyzes research papers and user projects using multiple AI models via API calls. The app provides comprehensive methodology extraction, comparative analysis, and detailed scoring with actionable recommendations.
![demo1](https://github.com/user-attachments/assets/97d87ea8-c979-498a-89ad-31f1b811e083)
![demo2](https://github.com/user-attachments/assets/1998b2d1-3632-4c97-91d2-ff764f9f95be)
## ✨ Features


### 🔗 **5-Step Analysis Chain**
1. **Document Structure Analysis** - Identifies abstract, methodology, and research objectives
2. **Deep Methodology Extraction** - Multi-pass extraction for completeness
3. **User Project Analysis** - Structured analysis of your research project
4. **Comparative Analysis** - Detailed comparison between paper and user approach
5. **Final Scoring & Recommendations** - Comprehensive evaluation with actionable insights

### 🤖 **Multiple AI Models**
- **🚀 Groq** - Fastest response times, free tier available
- **🤖 Gemini** - High quality responses, Google's latest AI
- **🏠 Ollama** - Completely local, no API limits

### 📊 **Advanced Features**
- **PDF Processing** - Extract text from research papers
- **Smart Token Management** - Optimized for each AI model
- **Fallback Systems** - Automatic recovery from API failures
- **Debug Panel** - Real-time monitoring and troubleshooting
- **Export Options** - Download results as TXT or PDF reports

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd reserch-analyst

# Install dependencies
pip install -r requirements.txt

# Install additional models (if needed)
python -m spacy download en_core_web_sm
```

### 2. **Setup AI Models**

#### **Groq (Recommended - Fastest)**
1. Get free API key from [console.groq.com](https://console.groq.com)
2. Enter API key in the sidebar
3. Test connection with the test button

#### **Gemini (High Quality)**
1. Get free API key from [aistudio.google.com](https://aistudio.google.com)
2. Enter API key in the sidebar

#### **Ollama (Local)**
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Run `ollama serve`
3. Pull model: `ollama pull llama3.2:1b`
4. Configure in sidebar

### 3. **Run the Application**
```bash
streamlit run main.py
```


## 🛠️ Troubleshooting

### **Common Issues**

#### **Groq Issues**
- **Token Limits**: Use shorter papers (under 10 pages)
- **Rate Limiting**: Wait between attempts
- **API Key**: Verify at console.groq.com

#### **Gemini Issues**
- **API Key**: Check at aistudio.google.com
- **Model Availability**: Try different Gemini models
- **Response Format**: App automatically cleans HTML tags

#### **Ollama Issues**
- **Timeout**: Use smaller models (1b, 3b)
- **Not Running**: Check `ollama serve`
- **Model Missing**: Run `ollama pull llama3.2:1b`

### **Performance Tips**
- **Groq**: Keep inputs under 1000 characters
- **Gemini**: Standard limits, good for complex papers
- **Ollama**: No limits but slower processing

## 📋 Requirements

### **Python Dependencies**
```
streamlit>=1.28.0
PyMuPDF>=1.23.0
requests>=2.31.0
sentence-transformers>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
spacy>=3.6.0
reportlab>=4.0.0
```

### **System Requirements**
- Python 3.8+
- 8GB+ RAM (for Ollama)
- Internet connection (for Groq/Gemini)


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


## Acknowledgments

- **Groq** for fast AI inference
- **Google** for Gemini AI
- **Ollama** for local AI processing
- **Streamlit** for the web framework

---

**💡 Tip**: Start with Groq for the fastest experience, then try Gemini for more detailed analysis, or Ollama for completely local processing.
