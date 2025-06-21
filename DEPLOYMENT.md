# Deployment Guide

This guide will help you deploy the AI Research Analyst Pro to Streamlit Cloud and other platforms.

## Streamlit Cloud Deployment

### Prerequisites

1. **GitHub Account**: You need a GitHub account to host your code
2. **Groq API Key**: Get a free API key from [console.groq.com](https://console.groq.com)
3. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/)

### Step-by-Step Deployment

#### 1. Prepare Your Repository

1. Create a new GitHub repository
2. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to: `main.py`
6. Click "Deploy!"

#### 3. Configure API Key

1. In your Streamlit Cloud dashboard, go to your app
2. Click on "Settings" (⚙️ icon)
3. Go to "Secrets" tab
4. Add your Groq API key:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```
5. Click "Save"
6. Your app will automatically redeploy

### Environment Variables

The app supports the following environment variables:

- `GROQ_API_KEY`: Your Groq API key (required)

### Troubleshooting

#### Common Issues

1. **Import Errors**: Make sure all dependencies are in `requirements.txt`
2. **Model Download Issues**: The spaCy model is downloaded automatically
3. **API Key Issues**: Check that your API key is correctly set in Streamlit secrets
4. **Memory Issues**: The app is optimized for Streamlit Cloud's memory limits

#### Debug Steps

1. Check the deployment logs in Streamlit Cloud
2. Verify your API key is working locally first
3. Ensure all files are committed to GitHub
4. Check that `main.py` is in the root directory

## Alternative Deployment Options

### Heroku

1. Create a `Procfile`:
```
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

2. Set environment variables in Heroku dashboard

### Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t research-analyst .
docker run -p 8501:8501 -e GROQ_API_KEY=your-key research-analyst
```

## Performance Optimization

### For Streamlit Cloud

1. **Caching**: The app uses `@st.cache_resource` for model loading
2. **Memory Management**: Large PDFs are processed in chunks
3. **API Limits**: Requests are optimized to stay within Groq's limits

### Monitoring

1. **Usage Tracking**: Monitor your Groq API usage
2. **Performance**: Check Streamlit Cloud metrics
3. **Errors**: Review deployment logs regularly

## Security Considerations

1. **API Key Protection**: Never commit API keys to version control
2. **Input Validation**: The app validates PDF uploads
3. **Rate Limiting**: Built-in retry logic for API calls

## Support

If you encounter issues:

1. Check the [Streamlit documentation](https://docs.streamlit.io/)
2. Review the [Groq API documentation](https://console.groq.com/docs)
3. Open an issue on GitHub
4. Check the deployment logs for specific error messages 