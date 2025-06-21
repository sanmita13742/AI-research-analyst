#!/usr/bin/env python3
"""
Setup script for AI Research Analyst Pro
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setting up AI Research Analyst Pro...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
        print("❌ Failed to download spaCy model")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Get a free Groq API key from: https://console.groq.com")
    print("2. Run the app: streamlit run main.py")
    print("3. Enter your API key in the sidebar")
    print("\n🌐 For deployment:")
    print("1. Push to GitHub")
    print("2. Deploy on Streamlit Cloud")
    print("3. Set GROQ_API_KEY in Streamlit secrets")

if __name__ == "__main__":
    main() 