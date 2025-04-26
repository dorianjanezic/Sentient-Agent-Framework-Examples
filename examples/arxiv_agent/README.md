# arXiv Research Agent

This agent uses the [Sentient Agent Framework](https://github.com/sentient-agi/Sentient-Agent-Framework) to provide a research assistant that can search and analyze papers from arXiv.

## Features

- Search arXiv for papers on specific topics
- Retrieve detailed information about papers by arXiv ID
- Generate summaries of papers
- Provide an overview of search results

## Setup and Installation

### 1. Create Environment File

Create the `.env` file by copying the contents of `.env.example`. This is where you will store your API credentials.

```bash
cp .env.example .env
```

Then edit the `.env` file to add your API keys:

```
MODEL_API_KEY="your-api-key"
MODEL_BASE_URL="https://api.openai.com/v1"  # Change if using a different provider
MODEL_NAME="gpt-4"  # Change to your preferred model
```

### 2. Create Python Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Agent

```bash
python -m src.arxiv_agent.arxiv_agent
```

## Usage

You can query the agent using tools like cURL or Postman:

```bash
curl -N --location 'http://0.0.0.0:8000/assist' \
--header 'Content-Type: application/json' \
--data '{
    "query": {
        "id": "unique_query_id",
        "prompt": "Find papers about large language models"
    },
    "session" : {
        "processor_id": "Example processor ID",
        "activity_id": "activity_id",
        "request_id": "request_id",
        "interactions": []
    }
}'
```

## Example Queries

- General search: "Find papers about quantum computing algorithms"
- Specific paper: "Show me details for paper 2311.12399"
- Topic exploration: "What are the latest papers on diffusion models?"

## Future Enhancements

- Add literature review capabilities
- Implement trend analysis
- Add cross-domain connection discovery
- Create personalized research feeds