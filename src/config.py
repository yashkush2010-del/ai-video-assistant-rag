import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required configuration
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set.")