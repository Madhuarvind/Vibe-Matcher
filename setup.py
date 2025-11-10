#!/usr/bin/env python3
"""
Setup script for Vibe Matcher project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def check_gemini_key():
    """Check if Gemini API key is configured"""
    # API key is hardcoded in the notebook
    GEMINI_API_KEY = 'AIzaSyDf8-vSXxw7g68RF4cdARIIikeERFG7G94'
    if GEMINI_API_KEY and len(GEMINI_API_KEY) > 20:
        print(f"✅ Gemini API key configured: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:]}")
        return True
    else:
        print("⚠️  Gemini API key not properly configured!")
        return False

def main():
    print("🚀 Setting up Vibe Matcher project...")
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check API key
    check_gemini_key()
    
    print("\n🎉 Setup complete!")
    print("Run: jupyter notebook vibe_matcher_notebook.ipynb")

if __name__ == "__main__":
    main()
