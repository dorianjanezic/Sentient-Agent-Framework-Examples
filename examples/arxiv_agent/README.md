# arXiv Research Agent

The arXiv Research Agent helps you explore scientific papers through natural language. It finds relevant papers, analyzes author profiles, provides paper summaries, and answers follow-up questions about the research.

## Key Features

### 1. Intelligent Paper Search and Discovery
- Advanced semantic search capabilities across arXiv's comprehensive database
- Smart query understanding with automatic refinement and enhancement
- Multi-dimensional filtering options:
  - Author-based filtering with publication history
  - Category-specific searches across all major academic disciplines
  - Date range filtering for tracking research evolution
  - Customizable sorting by relevance, submission date, or last update
- Intelligent search result grouping and thematic organization

### 2. Comprehensive Paper Analysis and Understanding
- Detailed paper summaries with key findings and contributions
- Author profiling with publication history and research trajectory
- Category-specific insights and academic context
- Cross-referencing with related research and citations
- Automatic identification of research trends and patterns
- Historical context and evolution of research topics

### 3. Interactive Research Assistant
- Natural language interaction with context-aware responses
- Intelligent follow-up question handling
- Personalized research recommendations
- Multi-turn conversation support
- Session-based context management
- Academic language understanding and response generation

### 4. Advanced Academic Features
- Comprehensive support for all major arXiv categories:
  - Computer Science (AI, ML, CV, NLP, etc.)
  - Mathematics (Algebra, Analysis, Topology, etc.)
  - Physics (Quantum, Condensed Matter, High Energy, etc.)
  - Economics, Quantitative Finance, and Statistics
  - Electrical Engineering and Systems Science
  - Astrophysics and Cosmology
  - And many more specialized fields
- Academic citation tracking and reference management
- Journal reference and DOI information
- Direct PDF access and paper retrieval
- Research trend analysis and pattern recognition

### 5. Technical Excellence
- Real-time arXiv API integration with optimized performance
- Advanced query classification and processing
- Robust error handling and fallback mechanisms
- Scalable architecture for research workloads
- Session-based context management
- Intelligent response generation with academic precision
- Automatic content filtering and quality control

## Technical Capabilities

- Real-time arXiv API integration
- Advanced query classification and processing
- Contextual session management
- Intelligent response generation
- Error handling and fallback mechanisms
- Scalable architecture for research workloads

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

You can interact with the agent using tools like cURL or Postman:

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

1. General Research Search:
   - "Find recent papers about quantum computing algorithms"
   - "Show me papers about transformer architectures in NLP"
   - "What are the latest developments in reinforcement learning?"

2. Specific Paper Queries:
   - "Show me details for paper 2311.12399"
   - "What's the main contribution of paper 2203.02155?"
   - "Find papers by author John Smith"

3. Topic Exploration:
   - "What are the latest papers on diffusion models?"
   - "Show me papers about AI safety and alignment"
   - "Find research on quantum machine learning"

4. Follow-up Questions:
   - "Tell me more about the first paper"
   - "What's the connection between these papers?"
   - "Who are the main authors in this field?"

## Future Enhancements

- Literature review generation
- Research trend analysis and visualization
- Cross-domain connection discovery
- Personalized research feeds
- Citation network analysis
- Research impact assessment
- Conference paper tracking
- Research collaboration suggestions

## Contributing

We welcome contributions to improve the arXiv Research Agent. Please feel free to submit issues, feature requests, or pull requests to help make this tool more useful for the research community.