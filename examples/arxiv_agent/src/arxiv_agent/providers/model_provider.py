from datetime import datetime
import asyncio
import logging
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError
from typing import AsyncIterator, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ModelProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"
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
        
        # Unified system prompt template
        self.system_prompt_template = """
        You are a professional academic research assistant specializing in scientific papers from arXiv, designed to engage researchers in thoughtful, open-ended discussions and provide precise, tool-based assistance when requested. Today is {date_today}. Your role is to assist across disciplines (e.g., Physics, Computer Science, Biology) by:

        1. Facilitating academic discussions on concepts, trends, or questions with clear, concise explanations.
        2. Connecting discussions to relevant research only when appropriate or requested.
        3. Performing specific tasks (e.g., searching papers, summarizing papers, profiling authors) when explicitly indicated.

        General Guidelines:
        - Use formal academic language suitable for PhD-level researchers, maintaining a conversational yet professional tone.
        - Be objective, precise, and concise, avoiding speculation or unsupported claims.
        - Avoid informal language, slang, colloquialisms, or profanity under any circumstances.
        - Structure responses clearly (e.g., numbered lists, sections) for readability.
        - Acknowledge limitations if information is unavailable or unclear.
        - Encourage users to continue the conversation, refine queries, or explore related research optionally.
        - For discussions, provide context on research trends or interdisciplinary connections when relevant.
        - For tool-based tasks, follow the specified instructions precisely.

        Supported Tasks:
        - Discuss or explain academic concepts (e.g., definitions, applications, trends).
        - Search for papers by topic, author, or category.
        - Summarize papers, focusing on contributions, methodology, or findings.
        - Generate author profiles with publication trends and collaborators.
        - Provide information about the agent's capabilities.
        - Classify user queries to select appropriate tools or discussion modes.

        Task-Specific Instructions:
        {task_instructions}
        """
        
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
        user_prompt: str,
        task_instructions: str = "",
        retry_count: int = 0,
        max_retries: int = 2
    ) -> AsyncIterator[str]:
        """
        Sends query to model and yields the response in chunks.
        
        Args:
            user_prompt: The query text to send to the model
            task_instructions: Task-specific instructions for the system prompt
            retry_count: Current retry attempt (internal use)
            max_retries: Maximum number of retries
            
        Yields:
            Response text chunks
        """
        # Format system prompt with current date and task instructions
        system_prompt = self.system_prompt_template.format(
            date_today=self.date_context,
            task_instructions=task_instructions
        )
        
        # Select model based on retry count (fallback to alternate models if needed)
        current_model = self.model
        if retry_count > 0 and retry_count <= len(self.fallback_models):
            current_model = self.fallback_models[retry_count - 1]
            logger.info(f"Using fallback model: {current_model}")

        # Prepare message for Fireworks AI format (system instructions in user message)
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
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
                async for chunk in self.query_stream(user_prompt, task_instructions, retry_count + 1, max_retries):
                    yield chunk
            else:
                logger.error(f"Max retries exceeded: {str(e)}")
                yield "I apologize, but I'm experiencing technical difficulties. Please try again later."
                raise

    async def query(
        self,
        user_prompt: str,
        task_instructions: str = "",
        max_retries: int = 2
    ) -> str:
        """
        Sends query to model and returns the complete response as a string.
        
        Args:
            user_prompt: The query text to send to the model
            task_instructions: Task-specific instructions for the system prompt
            max_retries: Maximum number of retries
            
        Returns:
            Complete response as a string
        """
        try:
            chunks = []
            async for chunk in self.query_stream(
                user_prompt=user_prompt, 
                task_instructions=task_instructions, 
                max_retries=max_retries
            ):
                chunks.append(chunk)
            response = "".join(chunks)
            return response
        except (RateLimitError, APIError, APITimeoutError) as e:
            logger.error(f"Query failed after max retries: {str(e)}")
            # Return a basic response if all retries fail
            return "I experienced technical difficulties. Please try again later."
        
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
        # Task-specific instructions for paper summarization
        task_instructions = """
        Summarize scientific papers, focusing on key contributions, methodology, and findings.
        Tailor the summary length (short: ~100 words, medium: ~200 words, long: ~300 words).
        Structure the summary for academic clarity.
        """
        
        # Create prompt for summarization
        user_prompt = f"""
        Summarize the following paper in a {length} format:
        
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Published: {paper['published']}
        Categories: {', '.join(paper['categories'])}
        Abstract: {paper['summary']}
        
        Include:
        1. Key contributions
        2. Methodology
        3. Main findings
        """
        
        try:
            # Try to generate the summary with retries
            async for chunk in self.query_stream(
                user_prompt, 
                task_instructions,
                max_retries=max_retries
            ):
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
            Due to processing limitations, a detailed summary is unavailable.
            The abstract begins: "{paper['summary'][:150]}..."
            Access the full paper at: {paper['arxiv_url']}
            """
            
            # Yield the fallback summary in chunks to simulate streaming
            for line in fallback_summary.split('\n'):
                yield line + '\n'
                await asyncio.sleep(0.05)  # Small delay to simulate streaming