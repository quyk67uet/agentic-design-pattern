import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4"  # or your preferred model
OPENAI_TEMPERATURE = 0

# Model Configuration
CNN_CONFIG = {
    "input_channels": 1,
    "num_classes": 10,
    "hidden_size": 16
}

MLP_CONFIG = {
    "input_size": 784,
    "hidden_size": 128,
    "num_classes": 10
}

# Image Configuration
IMAGE_SIZE = 28  # Assuming 28x28 images 