from datetime import datetime
import asyncio
import logging
from langchain_core.prompts import PromptTemplate
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError
from typing import AsyncIterator, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ModelProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        model: str = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
    ):
        """
        Initialize the model provider.
        
        Args:
            api_key: API key for the model provider
            base_url: Base URL for the API (default is Fireworks AI)
            model: Model identifier to use
        """
        # Model provider API key
        self.api_key = api_key
        # Model provider URL
        self.base_url = base_url
        # Identifier for specific model that should be used
        self.model = model
        # Temperature setting for response randomness
        self.temperature = 0.2
        # Maximum number of tokens for responses
        self.max_tokens = None
        
        # Current date for context
        self.date_context = datetime.now().strftime("%Y-%m-%d")
        
        # System prompt template
        self.system_prompt = (
    "You are a professional research assistant specializing in scientific papers from arXiv. "
    "Today is {date_today}. Your role is to help researchers discover, understand, and analyze "
    "scientific literature across a wide range of disciplines. "
    "\n\n"
    "You have expertise in the following major research areas:"
    "\n- Physics: including quantum mechanics, particle physics, astrophysics, and condensed matter"
    "\n- Mathematics: covering pure and applied mathematics, statistics, and mathematical physics"
    "\n- Computer Science: including AI, machine learning, algorithms, and systems"
    "\n- Engineering: spanning electrical, mechanical, civil, and aerospace engineering"
    "\n- Biology: covering molecular biology, genetics, neuroscience, and bioinformatics"
    "\n- Chemistry: including physical, organic, and computational chemistry"
    "\n- Economics: covering theoretical and applied economics, econometrics"
    "\n- Quantitative Finance: including mathematical finance and risk management"
    "\n- Statistics: covering probability theory, statistical methods, and data analysis"
    "\n- Quantitative Biology: including systems biology and biophysics"
    "\n- Quantitative Finance: covering financial mathematics and market analysis"
    "\n- Computer Science: including artificial intelligence, machine learning, and systems"
    "\n- Electrical Engineering and Systems Science: covering signal processing and control systems"
    "\n- Mathematics: including algebra, analysis, geometry, and topology"
    "\n- Physics: covering classical and quantum physics, astrophysics, and cosmology"
    "\n- Statistics: including probability theory and statistical methods"
    "\n\n"
    "When responding, follow these guidelines:"
    "\n- Provide concise, accurate information using formal academic language"
    "\n- Be objective and precise in your descriptions of research"
    "\n- Highlight key contributions and methodologies from papers"
    "\n- Connect papers through common themes, methods, or findings when relevant"
    "\n- Maintain a scholarly tone appropriate for academic discussion"
    "\n- Avoid informal language, colloquialisms, or extreme expressions"
    "\n- When uncertain, acknowledge limitations rather than speculating"
    "\n- Consider interdisciplinary connections between different fields"
    "\n- Provide context about how research fits into broader scientific trends"
    "\n- Highlight both theoretical and practical implications of research"
    "\n\n"
    "Remember that users are typically looking for recent and relevant research, "
    "so prioritize papers that match both their query and recency when appropriate. "
    "Be prepared to help users explore both established and emerging research areas, "
    "and to make connections between different fields of study."
)
        
        # Format system prompt with current date
        self.system_prompt = self.system_prompt.format(date_today=self.date_context)

        # Set up model API
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # Add a backup model for fallback
        self.fallback_models = [
            "accounts/fireworks/models/llama-v3-70b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct"
        ]

    async def query_stream(
        self,
        query: str,
        system_prompt: str = None,
        retry_count: int = 0,
        max_retries: int = 2
    ) -> AsyncIterator[str]:
        """
        Sends query to model and yields the response in chunks.
        
        Args:
            query: The query text to send to the model
            system_prompt: Optional custom system prompt
            retry_count: Current retry attempt (internal use)
            max_retries: Maximum number of retries
            
        Yields:
            Response text chunks
        """
        # Use provided system prompt or default
        current_system_prompt = system_prompt or self.system_prompt
        
        # Select model based on retry count (fallback to alternate models if needed)
        current_model = self.model
        if retry_count > 0 and retry_count <= len(self.fallback_models):
            current_model = self.fallback_models[retry_count - 1]
            logger.info(f"Using fallback model: {current_model}")

        # Prepare message format based on model type
        if "accounts/" in current_model or current_model in ["o1-preview", "o1-mini"]:
            # For models that need special formatting (Claude, Fireworks, etc.)
            messages = [
                {"role": "user",
                 "content": f"System Instruction: {current_system_prompt} \n Instruction: {query}"}
            ]
        else:
            # Standard OpenAI format
            messages = [
                {"role": "system", "content": current_system_prompt},
                {"role": "user", "content": query}
            ]

        try:
            # Create streaming completion request
            stream = await self.client.chat.completions.create(
                model=current_model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Yield response chunks
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except (RateLimitError, APIError, APITimeoutError) as e:
            # Handle rate limits and API errors with retries
            if retry_count < max_retries:
                logger.warning(f"API error: {str(e)}. Retrying ({retry_count + 1}/{max_retries})...")
                # Exponential backoff
                await asyncio.sleep(2 ** retry_count)
                async for chunk in self.query_stream(query, system_prompt, retry_count + 1, max_retries):
                    yield chunk
            else:
                logger.error(f"Max retries exceeded: {str(e)}")
                yield f"I apologize, but I'm currently experiencing technical difficulties. "
                yield f"Here's a basic response to your query: "
                
                # Generate a simple fallback response
                simple_response = f"Your query was about {query.split()[:5]}... " 
                simple_response += "I found relevant information but am unable to provide detailed analysis at this moment."
                yield simple_response
                raise

    async def query(
        self,
        query: str,
        system_prompt: str = None,
        max_retries: int = 2
    ) -> str:
        """
        Sends query to model and returns the complete response as a string.
        
        Args:
            query: The query text to send to the model
            system_prompt: Optional custom system prompt
            max_retries: Maximum number of retries
            
        Returns:
            Complete response as a string
        """
        try:
            chunks = []
            async for chunk in self.query_stream(query=query, system_prompt=system_prompt, max_retries=max_retries):
                chunks.append(chunk)
            response = "".join(chunks)
            return response
        except (RateLimitError, APIError, APITimeoutError) as e:
            logger.error(f"Query failed after max retries: {str(e)}")
            # Return a basic response if all retries fail
            return f"I processed your query about {query.split()[:5]}... but am currently experiencing technical limitations. Please try again later."
        
    async def summarize_paper(
        self,
        paper: Dict[str, Any],
        length: str = "medium",
        max_retries: int = 2
    ) -> AsyncIterator[str]:
        """
        Generate a summary of a paper.
        
        Args:
            paper: Paper metadata dictionary
            length: Summary length ("short", "medium", or "long")
            max_retries: Maximum retry attempts
            
        Yields:
            Summary text chunks
        """
        # Create prompt for summarization
        summary_prompt = f"""
        Please provide a {length} summary of the following scientific paper using formal academic language:
        
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Published: {paper['published']}
        Categories: {', '.join(paper['categories'])}
        
        Abstract:
        {paper['summary']}
        
        Your summary should:
        1. Focus on the key contributions, methodology, and findings
        2. Use formal language appropriate for an academic conference
        3. Be objective and precise in describing the research
        4. Avoid any informal expressions, colloquialisms, or profanity
        5. Maintain a professional tone throughout
        """
        
        # Custom system prompt for summarization
        system_prompt = (
            "You are a professional scientific research assistant with expertise in summarizing "
            "academic papers. Your summaries are clear, accurate, and use formal scholarly language. "
            "You write at a level appropriate for PhD-level researchers and maintain objectivity "
            "and precision in your analyses. Never use casual language, slang, or profanity."
        )
        
        try:
            # Try to generate the summary with retries
            async for chunk in self.query_stream(summary_prompt, system_prompt, max_retries=max_retries):
                yield chunk
        except (RateLimitError, APIError, APITimeoutError):
            # Provide a basic fallback summary if we exhaust all retries
            title = paper['title']
            authors = ', '.join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors += " et al."
            
            fallback_summary = f"""
Summary of "{title}" by {authors} ({paper['published']}):

This paper falls within the categories of {', '.join(paper['categories'])}.

Due to current processing limitations, I can only provide this basic information.
The paper's abstract begins with: "{paper['summary'][:150]}..."

For more details, you may want to access the full paper at: {paper['arxiv_url']}
            """
            
            # Yield the fallback summary in chunks to simulate streaming
            for line in fallback_summary.split('\n'):
                yield line + '\n'
                await asyncio.sleep(0.05)  # Small delay to simulate streaming