import logging
import os
import re
import asyncio
import random
from dotenv import load_dotenv
from typing import AsyncIterator, List, Dict, Any
from openai import RateLimitError, APIError, APITimeoutError

# Import providers
from src.arxiv_agent.providers.model_provider import ModelProvider
from src.arxiv_agent.providers.arxiv_provider import ArxivProvider

# Import Sentient Agent Framework components
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler
)

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ArxivResearchAgent(AbstractAgent):
    def __init__(
            self,
            name: str
    ):
        """Initialize the arXiv research agent."""
        super().__init__(name)

        # Initialize model provider
        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            raise ValueError("MODEL_API_KEY is not set")
        
        model_base_url = os.getenv("MODEL_BASE_URL", "https://api.fireworks.ai/inference/v1")
        model_name = os.getenv("MODEL_NAME", "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new")
        
        self._model_provider = ModelProvider(
            api_key=model_api_key,
            base_url=model_base_url,
            model=model_name
        )

        # Initialize arXiv provider
        self._arxiv_provider = ArxivProvider()

    async def assist(
            self,
            session: Session,
            query: Query,
            response_handler: ResponseHandler
    ):
        """
        Process the user query and provide research assistance.
        
        Args:
            session: Session information
            query: User query
            response_handler: Handler for sending responses
        """
        user_query = query.prompt.lower()  # Convert to lowercase for easier matching
        
        # Initialize session metadata if it doesn't exist
        if not hasattr(session, "metadata") or session.metadata is None:
            session.metadata = {}
        
        # Log the current session state for debugging
        if hasattr(session, "metadata") and session.metadata:
            paper_count = len(session.metadata.get("last_papers", []))
            last_query = session.metadata.get("last_query", "None")
            logger.info(f"Session has metadata. Papers: {paper_count}, Last query: {last_query}")
        else:
            logger.info("Session has no metadata or is empty")
        
        try:
            # Check if this is a follow-up request about a specific paper
            follow_up_phrases = ["this paper", "that paper", "the paper", "more details", "tell me more", 
                                "elaborate", "full text", "download", "pdf", "details"]
            
            is_follow_up = any(phrase in user_query for phrase in follow_up_phrases)
            has_last_papers = hasattr(session, "metadata") and "last_papers" in session.metadata
            
            if is_follow_up:
                logger.info(f"Detected potential follow-up query: '{query.prompt}'")
                
            if is_follow_up and has_last_papers and len(session.metadata["last_papers"]) > 0:
                logger.info("Handling as follow-up query about a specific paper")
                # Get the most relevant paper from the previous search
                most_relevant_paper = session.metadata["last_papers"][0]
                
                # Extract paper ID safely
                paper_id = None
                if "id" in most_relevant_paper:
                    paper_id_full = most_relevant_paper["id"]
                    logger.info(f"Raw paper ID: {paper_id_full}")
                    
                    # Handle different ID formats
                    if "/" in paper_id_full:
                        paper_id = paper_id_full.split("/")[-1]
                    else:
                        paper_id = paper_id_full
                        
                    # Remove version if present
                    if paper_id and "v" in paper_id:
                        paper_id = paper_id.split("v")[0]
                        
                    logger.info(f"Extracted paper ID: {paper_id}")
                
                if paper_id:
                    await self._handle_paper_detail(paper_id, response_handler, session)
                    return
                else:
                    logger.error("Failed to extract paper ID from the most relevant paper")
            
            # If not a follow-up or no last papers, proceed with normal handling
            # Check if the query contains an arXiv ID or URL
            arxiv_id_pattern = r'(\d{4}\.\d{5})(v\d+)?'
            arxiv_url_pattern = r'arxiv\.org\/abs\/(\d{4}\.\d{5})(v\d+)?'
            
            arxiv_id_match = re.search(arxiv_id_pattern, query.prompt)
            arxiv_url_match = re.search(arxiv_url_pattern, query.prompt)
            
            if arxiv_id_match or arxiv_url_match:
                # Extract paper ID
                paper_id = arxiv_id_match.group(1) if arxiv_id_match else arxiv_url_match.group(1)
                await self._handle_paper_detail(paper_id, response_handler, session)
            else:
                # Handle as search query
                await self._handle_search_query(query.prompt, response_handler, session)
        except Exception as e:
            # Global error handler to ensure we always return something
            logger.error(f"Unexpected error in assist: {str(e)}")
            await response_handler.emit_text_block(
                "ERROR", "I encountered an unexpected error. Please try again later."
            )
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(
                "I apologize, but I experienced a technical issue while processing your request. "
                "Please try again with a different query or wait a moment before trying again."
            )
            await final_response_stream.complete()
            await response_handler.complete()
    
    async def _handle_search_query(
            self,
            query: str,
            response_handler: ResponseHandler,
            session: Session
    ):
        """
        Handle a search query by searching arXiv and returning results.
        
        Args:
            query: Search query
            response_handler: Handler for sending responses
            session: Session information for maintaining context
        """
        # Notify user that search is in progress
        await response_handler.emit_text_block(
            "SEARCH", "Searching arXiv for relevant papers..."
        )
        
        # Perform search
        search_results = await self._arxiv_provider.search(query, max_results=5)
        
        # Emit search results as JSON
        if search_results:
            # Store papers in session for context in future queries
            # Make sure we're storing the complete paper objects with all fields
            session.metadata["last_papers"] = search_results
            session.metadata["last_query"] = query
            
            # Log that we stored papers in the session
            logger.info(f"Stored {len(search_results)} papers in session context for query: '{query}'")
            if len(search_results) > 0:
                first_paper = search_results[0]
                paper_id = first_paper.get("id", "unknown")
                paper_title = first_paper.get("title", "unknown")
                logger.info(f"Most relevant paper: ID={paper_id}, Title={paper_title}")
            
            await response_handler.emit_json(
                "PAPERS", {"results": search_results}
            )
            
            try:
                # Create summary of search results
                summary_stream = response_handler.create_text_stream("SEARCH_SUMMARY")
                async for chunk in self._generate_search_summary(query, search_results):
                    await summary_stream.emit_chunk(chunk)
                await summary_stream.complete()
            except (RateLimitError, APIError, APITimeoutError) as e:
                # Handle model API errors in summarization
                logger.warning(f"Model API error during search summary generation: {str(e)}")
                await response_handler.emit_text_block(
                    "API_LIMIT", "The search summary service is currently experiencing high demand. Basic results are still available."
                )
            
            # Create final response stream - this is what the frontend expects
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            try:
                # Try to get the AI-generated summary
                summary = await self._generate_final_summary(query, search_results)
                await final_response_stream.emit_chunk(summary)
            except Exception as e:
                # Fallback to a simple summary if any error occurs
                logger.error(f"Error generating final summary: {str(e)}")
                paper_titles = [p.get('title', 'Untitled Paper') for p in search_results[:3]]
                fallback_response = f"""
I found {len(search_results)} papers related to your query "{query}".

The most relevant papers include:
- {paper_titles[0]}
{f"- {paper_titles[1]}" if len(paper_titles) > 1 else ""}
{f"- {paper_titles[2]}" if len(paper_titles) > 2 else ""}

You can ask for more details about any of these papers.
                """
                await final_response_stream.emit_chunk(fallback_response)
                
            await final_response_stream.complete()
        else:
            # No results found
            await response_handler.emit_text_block(
                "NO_RESULTS", "No papers found matching your query on arXiv."
            )
            
            # Still need to emit a FINAL_RESPONSE even when no results are found
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(
                f"I could not find any papers on arXiv matching your query: '{query}'. "
                f"Please try a different search term or check if your query is too specific."
            )
            await final_response_stream.complete()
            
            # Clear any previous papers since this search failed
            if "last_papers" in session.metadata:
                del session.metadata["last_papers"]
        
        # Mark response as complete
        await response_handler.complete()
    
    async def _handle_paper_detail(
            self,
            paper_id: str,
            response_handler: ResponseHandler,
            session: Session
    ):
        """
        Handle a request for details about a specific paper.
        
        Args:
            paper_id: arXiv paper ID
            response_handler: Handler for sending responses
            session: Session information for maintaining context
        """
        # Log that we're fetching a specific paper
        logger.info(f"Fetching details for paper with ID: {paper_id}")
        
        # Notify user that we're fetching the paper
        await response_handler.emit_text_block(
            "FETCH_PAPER", f"Fetching details for paper {paper_id}..."
        )
        
        # Get paper details
        try:
            paper = await self._arxiv_provider.get_paper_by_id(paper_id)
            logger.info(f"Successfully retrieved paper: {paper.get('title', 'Unknown title')}")
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {str(e)}")
            paper = {"error": f"Failed to retrieve paper: {str(e)}"}
        
        if "error" in paper:
            # Paper not found
            await response_handler.emit_text_block(
                "ERROR", paper["error"]
            )
            
            # Still need to emit a FINAL_RESPONSE even when paper is not found
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(
                f"I could not find the paper with ID {paper_id} on arXiv. "
                f"Please check if the ID is correct and try again."
            )
            await final_response_stream.complete()
            
            await response_handler.complete()
            return
        
        # Store the current paper in session for future reference
        session.metadata["current_paper"] = paper
        
        # Emit paper details as JSON
        await response_handler.emit_json(
            "PAPER_DETAILS", {"paper": paper}
        )
        
        try:
            # Generate and stream paper summary
            summary_stream = response_handler.create_text_stream("PAPER_SUMMARY")
            async for chunk in self._model_provider.summarize_paper(paper, length="medium"):
                await summary_stream.emit_chunk(chunk)
            await summary_stream.complete()
        except (RateLimitError, APIError, APITimeoutError) as e:
            # Handle API errors during summary generation
            logger.warning(f"API error during paper summary generation: {str(e)}")
            await response_handler.emit_text_block(
                "API_LIMIT", "I'm unable to generate a detailed summary at this time due to high demand."
            )
        
        # Determine if this was a follow-up query
        is_followup = False
        if "last_papers" in session.metadata:
            last_papers = session.metadata["last_papers"]
            is_followup = any(p["id"].split("/")[-1].split("v")[0] == paper_id for p in last_papers)
        
        # Create final response stream - this is what the frontend expects
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        # Create a concise final response about the paper
        title = paper['title']
        authors = ', '.join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors += " et al."
        published = paper['published']
        categories = ', '.join(paper['categories'])
        summary = paper.get('summary', '').strip()
        
        # Prepare a more detailed response for follow-up queries
        if is_followup and "last_query" in session.metadata:
            final_response = f"""
Here are the detailed information about the paper related to your search for "{session.metadata['last_query']}":

Title: {title}
Authors: {authors}
Published: {published}
Categories: {categories}

Abstract:
{summary}

The paper is accessible at {paper['arxiv_url']} with the PDF available at {paper['pdf_url']}.

This paper was selected as the most relevant result from your previous search.
            """
        else:
            final_response = f"""
I found the paper you requested (ID: {paper_id}):

Title: {title}
Authors: {authors}
Published: {published}
Categories: {categories}

Abstract:
{summary}

The paper is accessible at {paper['arxiv_url']} with a PDF available at {paper['pdf_url']}.
            """
        
        await final_response_stream.emit_chunk(final_response)
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
    
    async def _generate_search_summary(
            self,
            query: str,
            papers: List[Dict[str, Any]]
    ) -> AsyncIterator[str]:
        """
        Generate a summary of search results.
        
        Args:
            query: Original search query
            papers: List of paper metadata
            
        Yields:
            Summary text chunks
        """
        # Create a prompt for summarizing the search results
        papers_text = "\n\n".join([
            f"Title: {paper['title']}\n"
            f"Authors: {', '.join(paper['authors'])}\n"
            f"Published: {paper['published']}\n"
            f"Categories: {', '.join(paper['categories'])}\n"
            f"Summary: {paper['summary'][:200]}..."
            for paper in papers
        ])
        
        prompt = f"""
        Based on the search query: "{query}"
        
        I found these papers on arXiv:
        
        {papers_text}
        
        Please provide:
        1. A formal academic overview of these research papers (2-3 sentences per paper)
        2. How they relate to the search query (1-2 sentences)
        3. Which papers might be most relevant to the search query and why (2-3 sentences)
        
        Use professional academic language throughout. Do not use informal language or profanity.
        """
        
        # Custom system prompt for search summary
        system_prompt = (
            "You are a professional academic research assistant specializing in scientific literature. "
            "You help researchers understand available papers on topics using formal, scholarly language. "
            "Never use casual expressions, slang, or profanity in your analyses. Maintain a professional, "
            "objective tone suitable for publication in an academic journal."
        )
        
        try:
            # The previous approach tried to iterate directly over the retry function, which caused an error
            # Instead, we'll use a different approach to implement retries for streaming
            max_retries = 2
            retry_count = 0
            
            while True:
                try:
                    # Get a normal iterator from query_stream and iterate over it
                    async for chunk in self._model_provider.query_stream(prompt, system_prompt):
                        yield chunk
                    # If we get here, everything worked fine, so break the loop
                    break
                    
                except (RateLimitError, APIError, APITimeoutError) as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        # If we've exceeded retries, re-raise to be caught by the outer try/except
                        raise
                    
                    # Calculate backoff time
                    delay = 2 ** retry_count * (1 + 0.1 * random.random())
                    logger.warning(f"API error in search summary: {str(e)}. Retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                    
        except (RateLimitError, APIError, APITimeoutError):
            # If we still hit rate limits after retries, provide a simpler response
            simple_summary = f"""
Based on your query "{query}", I found {len(papers)} relevant papers.

Here is a brief overview:

"""
            yield simple_summary
            
            # Add a simple description of each paper
            for i, paper in enumerate(papers[:3]):  # Limit to top 3 papers
                paper_summary = f"""
Paper {i+1}: "{paper['title']}" by {', '.join(paper['authors'][:2])}
Published: {paper['published']}
Category: {paper['primary_category']}

"""
                yield paper_summary
            
            yield "\nYou can ask for more details about any specific paper you find interesting."
    
    async def _generate_final_summary(
            self,
            query: str,
            papers: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a concise final summary response for the user.
        
        Args:
            query: Original search query
            papers: List of paper metadata
            
        Returns:
            Final summary text
        """
        # Create a prompt for the final response
        prompt = f"""
        Based on the search query: "{query}"
        
        I need a concise, direct response to the user about the papers I found.
        Create a brief, professional final answer that:
        1. Acknowledges their query
        2. Mentions the number of papers found ({len(papers)})
        3. Highlights the most relevant paper(s)
        4. Provides a very brief summary of key findings
        5. Indicates that the user can ask for more details about a specific paper
        
        Be direct and helpful, using formal academic language. This should be a complete 
        standalone response to the user's query. Make sure to mention that they can ask for 
        "more details about this paper" to get additional information.
        """
        
        # Custom system prompt for final response
        system_prompt = (
            "You are a helpful, professional research assistant specializing in academic literature. "
            "Your responses are clear, concise, and academically appropriate. Always maintain "
            "a professional tone and be direct in addressing the user's needs."
        )
        
        try:
            # Using direct retry logic instead of the helper function
            max_retries = 3
            retry_count = 0
            
            while True:
                try:
                    response = await self._model_provider.query(prompt, system_prompt)
                    return response
                except (RateLimitError, APIError, APITimeoutError) as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise
                    
                    delay = 2 ** retry_count * (1 + 0.1 * random.random())
                    logger.warning(f"API error in final summary: {str(e)}. Retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                    
        except (RateLimitError, APIError, APITimeoutError):
            # Fallback response if we still encounter rate limits
            paper_titles = [p.get('title', 'Untitled Paper') for p in papers[:3]]
            return f"""
I found {len(papers)} papers related to your query "{query}".

The most relevant papers include:
- {paper_titles[0]}
{f"- {paper_titles[1]}" if len(paper_titles) > 1 else ""}
{f"- {paper_titles[2]}" if len(paper_titles) > 2 else ""}

You can ask for more details about any of these papers.

(Note: I'm currently experiencing high demand. Some details have been simplified.)
            """


if __name__ == "__main__":
    # Create an instance of the ArxivResearchAgent
    agent = ArxivResearchAgent(name="arXiv Research Agent")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()