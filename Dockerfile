FROM python:3.11-slim

WORKDIR /app

# Copy requirements file
COPY examples/arxiv_agent/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY examples/arxiv_agent/src/ ./src/

# Required environment variables
ENV MODEL_API_KEY=""
ENV MODEL_BASE_URL="https://api.fireworks.ai/inference/v1"
ENV MODEL_NAME="accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
ENV PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application with proper signal handling
CMD ["sh", "-c", "python -m src.arxiv_agent.arxiv_agent"] 