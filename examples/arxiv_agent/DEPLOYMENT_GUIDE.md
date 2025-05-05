# ArXiv Agent Deployment Guide

## Local Development and Deployment Process

### 1. Make Local Changes
- Make your code changes in the `src/` directory
- Test changes locally using:
  ```bash
  docker build -t arxiv-agent .
  docker run -p 8080:8080 -e MODEL_API_KEY="your-api-key" -e MODEL_BASE_URL="https://api.fireworks.ai/inference/v1" -e MODEL_NAME="accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new" arxiv-agent
  ```

### 2. Update Docker Image
1. Build the updated Docker image:
   ```bash
   docker build --platform linux/amd64 -t gcr.io/arxiv-agent-20250427191432/arxiv-agent .
   ```

2. Push the image to Google Container Registry:
   ```bash
   docker push gcr.io/arxiv-agent-20250427191432/arxiv-agent
   ```

### 3. Deploy to Google Cloud Run
1. Make sure you're authenticated with Google Cloud:
   ```bash
   gcloud auth login
   ```

2. Deploy the updated service:
   ```bash
   gcloud run deploy arxiv-agent \
     --image gcr.io/arxiv-agent-20250427191432/arxiv-agent \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars "MODEL_API_KEY=${MODEL_API_KEY},MODEL_BASE_URL=${MODEL_BASE_URL},MODEL_NAME=${MODEL_NAME}"
   ```

### 4. Verify Deployment
1. Check the deployment status:
   ```bash
   gcloud run services describe arxiv-agent --region us-central1
   ```

2. Test the updated service:
   ```bash
   curl -N --location 'https://arxiv-agent-b5n2q27qha-uc.a.run.app/assist' \
     --header 'Content-Type: application/json' \
     --data '{
       "query": {
         "id": "01JQETZTSNT4KC0TRS6EBN32TG",
         "prompt": "Your test query here"
       },
       "session": {
         "processor_id": "Example processor ID",
         "activity_id": "01JR8SXE9B92YDKKNMYHYFZY1T",
         "request_id": "01JR8SY5PHB9X2FET1QRXGZW76",
         "interactions": []
       }
     }'
   ```

### Environment Variables
Make sure these environment variables are set before deployment:
- `MODEL_API_KEY`: Your API key
- `MODEL_BASE_URL`: The model API base URL
- `MODEL_NAME`: The model name to use

### Troubleshooting
1. If deployment fails, check the logs:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=arxiv-agent" --limit 50
   ```

2. To rollback to a previous version:
   ```bash
   gcloud run services update-traffic arxiv-agent --to-revisions arxiv-agent-0000X-xxx=100
   ```
   (Replace X-xxx with the revision ID you want to rollback to)

### Current Service URL
- Production URL: https://arxiv-agent-b5n2q27qha-uc.a.run.app 
