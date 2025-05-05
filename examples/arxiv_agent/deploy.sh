#!/bin/bash
set -e

# Configuration
PROJECT_ID="arxiv-agent-20250427191432"
IMAGE_NAME="arxiv-agent"
REGION="us-central1"
SERVICE_NAME="arxiv-agent"

# Check if required environment variables are set
if [ -z "$MODEL_API_KEY" ]; then
    echo "Error: MODEL_API_KEY environment variable is not set"
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Push the image to Google Container Registry
echo "Pushing image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars "MODEL_API_KEY=${MODEL_API_KEY},MODEL_BASE_URL=${MODEL_BASE_URL},MODEL_NAME=${MODEL_NAME}"

echo "Deployment completed. Service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)" 