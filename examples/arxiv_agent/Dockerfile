FROM python:3.11-slim

WORKDIR /app

# Set Python path to include the src directory
ENV PYTHONPATH=/app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Required environment variables
ENV MODEL_API_KEY=""
ENV MODEL_BASE_URL="https://api.fireworks.ai/inference/v1"
ENV MODEL_NAME="accounts/fireworks/models/llama-v3-70b-instruct"

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "-m", "src.arxiv_agent.arxiv_agent"] 