import logging
import os
import re
import asyncio
import random
import json
import traceback
import uuid
from enum import Enum
from dotenv import load_dotenv
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple, Callable, Union

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

# Set up enhanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a stream handler if not already added
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Define query types for classification
class QueryType(Enum):
    SEARCH = "search"          # Search for papers on a topic
    PAPER_DETAIL = "detail"    # Get details about a specific paper
    FOLLOWUP = "followup"      # Follow-up question about previous results
    AGENT_INFO = "agent_info"  # Questions about the agent itself
    GREETING = "greeting"      # Hello, hi, etc.
    UNKNOWN = "unknown"        # Couldn't classify

# List of words to filter from model responses
INAPPROPRIATE_WORDS = [
    "fuck", "fucking", "fucked", "shit", "crap", "damn", "bitch", "ass", 
    "asshole", "bullshit", "dumb", "stupid", "idiot", "hell", "wtf", "bs",
    # Add more inappropriate words as needed
]

# Define tools that the agent can use
class Tools:
    @staticmethod
    async def search_papers(query: str, session: Session, agent: 'ArxivResearchAgent') -> Dict[str, Any]:
        """Search for papers on arXiv."""
        logger.info(f"[TOOL] search_papers called with query: '{query}'")
        
        # Ensure session metadata exists
        if not hasattr(session, "metadata") or session.metadata is None:
            session.metadata = {'client_id': 'default_user'}
            logger.info("[SESSION] Initialized empty session metadata")
        
        # Perform search
        try:
            search_results = await agent._arxiv_provider.search(query, max_results=5)
            logger.info(f"[TOOL] search_papers found {len(search_results)} results")
            
            # Log paper titles and IDs for debugging
            for i, paper in enumerate(search_results[:3]):
                logger.info(f"[TOOL] Paper {i+1}: '{paper.get('title')}' - ID: {paper.get('id')}")
            
            # Store in session for context in future queries
            if search_results:
                session.metadata["last_papers"] = search_results
                session.metadata["last_query"] = query
                session.metadata["last_action"] = "search"
                
                # Ensure we're storing the session metadata persistently
                await agent._update_session(session)
                
                logger.info(f"[SESSION] Stored {len(search_results)} papers in session")
                
                return {
                    "type": "search_results",
                    "query": query,
                    "count": len(search_results),
                    "results": search_results
                }
            else:
                logger.info("[TOOL] No search results found")
                return {
                    "type": "search_results",
                    "query": query,
                    "count": 0,
                    "results": []
                }
        except Exception as e:
            logger.error(f"[TOOL] Error in search_papers: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "type": "search_results",
                "query": query,
                "count": 0,
                "results": [],
                "error": str(e)
            }
    
    @staticmethod
    async def get_paper_details(paper_id: str, session: Session, agent: 'ArxivResearchAgent') -> Dict[str, Any]:
        """Get detailed information about a specific paper."""
        logger.info(f"[TOOL] get_paper_details called with paper_id: '{paper_id}'")
        
        # Ensure session metadata exists
        if not hasattr(session, "metadata") or session.metadata is None:
            session.metadata = {'client_id': 'default_user'}
            logger.info("[SESSION] Initialized empty session metadata")
        
        # Get paper details
        try:
            paper = await agent._arxiv_provider.get_paper_by_id(paper_id)
            logger.info(f"[TOOL] Successfully retrieved paper: '{paper.get('title')}'")
            
            # Store for future reference
            session.metadata["current_paper"] = paper
            session.metadata["last_action"] = "paper_detail"
            
            # Ensure we're storing the session metadata persistently
            await agent._update_session(session)
            
            logger.info(f"[SESSION] Stored current_paper in session")
            
            return {
                "type": "paper_detail",
                "paper_id": paper_id,
                "found": True,
                "paper": paper
            }
        except Exception as e:
            logger.error(f"[TOOL] Error fetching paper {paper_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "type": "paper_detail",
                "paper_id": paper_id,
                "found": False,
                "error": str(e)
            }
    
    @staticmethod
    async def get_agent_info(query: str, session: Session, agent: 'ArxivResearchAgent') -> Dict[str, Any]:
        """Return information about the agent itself."""
        logger.info(f"[TOOL] get_agent_info called with query: '{query}'")
        
        return {
            "type": "agent_info",
            "query": query,
            "name": agent.name,
            "description": "I am an arXiv Research Agent that can help you find and understand scientific papers. "
                        "I can search for papers on specific topics, provide details about individual papers, "
                        "and generate summaries of research findings.",
            "capabilities": [
                "Search for papers on arXiv by topic",
                "Retrieve details about specific papers by ID",
                "Generate summaries of research findings",
                "Provide academic context for search results"
            ]
        }


class ImprovedDefaultServer(DefaultServer):
    """An improved version of DefaultServer that maintains client sessions."""
    
    def __init__(self, agent: AbstractAgent):
        super().__init__(agent)
        self.client_sessions = {}
        logger.info("[SERVER] Initialized improved server with persistent client sessions")
    
    async def handle_request(self, request):
        """Handle incoming requests with proper session tracking."""
        
        # Extract client identifier - could be from headers, cookies, etc.
        client_id = request.headers.get('X-Client-ID')
        if not client_id:
            # Fall back to IP address if no client ID header
            client_id = f"ip_{request.client.host}"
        
        # Get or create client session
        if client_id in self.client_sessions:
            session = self.client_sessions[client_id]
            logger.info(f"[SERVER] Retrieved existing session for client {client_id}")
        else:
            session = Session()
            session.metadata = {'client_id': client_id}
            self.client_sessions[client_id] = session
            logger.info(f"[SERVER] Created new session for client {client_id}")
        
        # Set client ID in request context
        request_context = {'client_id': client_id, 'client_ip': request.client.host}
        
        # Process the request using the agent
        query = Query(prompt=request.json()['prompt'], context=request_context)
        
        # Create response handler
        response_handler = ResponseHandler()
        
        # Call agent's assist method
        await self.agent.assist(session, query, response_handler)
        
        # Store updated session
        self.client_sessions[client_id] = session
        
        return response_handler.get_response()


class ArxivResearchAgent(AbstractAgent):
    # Class-level session store to ensure it persists across instances
    _global_session_store = {}
    
    def __init__(
            self,
            name: str
    ):
        """Initialize the arXiv research agent."""
        super().__init__(name)
        logger.info(f"[INIT] Initializing ArxivResearchAgent with name: {name}")

        # Initialize model provider
        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            logger.error("[INIT] MODEL_API_KEY is not set")
            raise ValueError("MODEL_API_KEY is not set")
        
        model_base_url = os.getenv("MODEL_BASE_URL", "https://api.fireworks.ai/inference/v1")
        model_name = os.getenv("MODEL_NAME", "accounts/fireworks/models/llama-v3-70b-instruct")
        
        logger.info(f"[INIT] Using model: {model_name} from {model_base_url}")
        
        self._model_provider = ModelProvider(
            api_key=model_api_key,
            base_url=model_base_url,
            model=model_name
        )

        # Initialize arXiv provider
        self._arxiv_provider = ArxivProvider()
        
        # Use the class-level session store
        self._session_store = ArxivResearchAgent._global_session_store
        logger.info(f"[INIT] Using global session store with {len(self._session_store)} existing sessions")
        
        logger.info("[INIT] ArxivResearchAgent initialization complete")

    async def _update_session(self, session: Session) -> None:
        """Ensure session metadata is stored persistently."""
        if not hasattr(session, 'metadata') or not session.metadata:
            logger.warning("[SESSION] Cannot update session: no metadata found")
            return
            
        # Get client ID from session metadata
        client_id = session.metadata.get('client_id', 'default_user')
        
        # Store session metadata using client ID
        self._session_store[client_id] = session.metadata
        logger.info(f"[SESSION] Updated persistent session store for client {client_id}")

    def _filter_inappropriate_content(self, text: str) -> str:
        """Filter out inappropriate language from model responses."""
        if not text:
            return text
            
        filtered_text = text
        replaced_words = False
        
        for word in INAPPROPRIATE_WORDS:
            # Pattern to match whole words (with word boundaries)
            pattern = r'\b' + re.escape(word) + r'\b'
            
            # Check if the word exists in the text
            if re.search(pattern, filtered_text, re.IGNORECASE):
                # Replace with appropriate substitutions
                if word in ["fuck", "fucking", "fucked"]:
                    filtered_text = re.sub(pattern, "extremely", filtered_text, flags=re.IGNORECASE)
                elif word in ["shit", "crap", "bs", "bullshit"]:
                    filtered_text = re.sub(pattern, "material", filtered_text, flags=re.IGNORECASE)
                elif word in ["damn"]:
                    filtered_text = re.sub(pattern, "remarkably", filtered_text, flags=re.IGNORECASE)
                elif word in ["hell"]:
                    filtered_text = re.sub(pattern, "extremely", filtered_text, flags=re.IGNORECASE)
                elif word in ["dumb", "stupid", "idiot"]:
                    filtered_text = re.sub(pattern, "misguided", filtered_text, flags=re.IGNORECASE)
                else:
                    # Default replacement for other inappropriate words
                    filtered_text = re.sub(pattern, "[inappropriate]", filtered_text, flags=re.IGNORECASE)
                
                replaced_words = True
        
        if replaced_words:
            logger.warning("[FILTER] Inappropriate language detected and filtered in response")
            
        return filtered_text

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
        logger.info(f"[ASSIST] Received query: '{query.prompt}'")
        
        try:
            # Extract client identifier from the request
            # Look for a client ID in various places
            client_id = None
            
            # Try to get from query.context
            context = getattr(query, 'context', {})
            client_id = context.get('client_id')
            
            # Try to get from request headers if available
            headers = context.get('request_headers', {})
            if not client_id and headers:
                client_id = headers.get('X-Client-ID')
            
            # If not found, try query.metadata
            if not client_id and hasattr(query, 'metadata'):
                client_id = query.metadata.get('client_id')
            
            # If not found, try from session metadata
            if not client_id and hasattr(session, 'metadata') and session.metadata:
                client_id = session.metadata.get('client_id')
            
            # If still not found, try to get from the HTTP request IP address if available
            if not client_id and 'client_ip' in context:
                client_id = f"ip_{context.get('client_ip')}"
            
            # If still no client ID, use a consistent value for this session
            if not client_id:
                # Generate a unique session ID
                client_id = f"session_{uuid.uuid4()}"
            
            logger.info(f"[SESSION] Using client ID: {client_id}")
            
            # Initialize session metadata if needed
            if not hasattr(session, 'metadata') or session.metadata is None:
                session.metadata = {}
            
            # Store client ID in session metadata
            session.metadata['client_id'] = client_id
            
            # Retrieve session data from global store if it exists
            if client_id in self._session_store:
                # Merge session data to preserve client_id
                stored_metadata = self._session_store[client_id]
                for key, value in stored_metadata.items():
                    if key != 'client_id':  # Preserve our new client_id
                        session.metadata[key] = value
                logger.info(f"[SESSION] Retrieved session data for client {client_id}")
            else:
                logger.info(f"[SESSION] No existing session data for client {client_id}")
            
            # Log the current session state
            self._log_session_state(session)
            
            # Step 1: Classify the query to determine intent
            query_type, entities = await self._classify_query(query.prompt, session)
            logger.info(f"[CLASSIFY] Query type: {query_type.value}, Entities: {entities}")
            
            # Step 2: Route to appropriate tool based on query type
            if query_type == QueryType.SEARCH:
                logger.info("[ROUTING] Handling as SEARCH query")
                search_term = entities.get("search_term", query.prompt)
                logger.info(f"[ROUTING] Search term: '{search_term}'")
                result = await Tools.search_papers(search_term, session, self)
                await self._handle_search_response(result, response_handler, session)
                
            elif query_type == QueryType.PAPER_DETAIL:
                logger.info("[ROUTING] Handling as PAPER_DETAIL query")
                paper_id = entities.get("paper_id")
                logger.info(f"[ROUTING] Paper ID: '{paper_id}'")
                if paper_id:
                    result = await Tools.get_paper_details(paper_id, session, self)
                    await self._handle_paper_detail_response(result, response_handler, session)
                else:
                    logger.error("[ROUTING] No paper ID found in entities")
                    await self._handle_error_response("Unable to identify a paper ID in the query.", response_handler)
                
            elif query_type == QueryType.FOLLOWUP:
                logger.info("[ROUTING] Handling as FOLLOWUP query")
                # Determine what the follow-up is about based on previous action
                last_action = session.metadata.get("last_action")
                logger.info(f"[ROUTING] Last action: '{last_action}'")
                
                if last_action == "search" and "last_papers" in session.metadata and len(session.metadata["last_papers"]) > 0:
                    # Follow-up about the most relevant paper from previous search
                    most_relevant_paper = session.metadata["last_papers"][0]
                    logger.info(f"[ROUTING] Most relevant paper: '{most_relevant_paper.get('title')}'")
                    
                    paper_id = self._extract_paper_id(most_relevant_paper)
                    logger.info(f"[ROUTING] Extracted paper ID: '{paper_id}'")
                    
                    if paper_id:
                        result = await Tools.get_paper_details(paper_id, session, self)
                        await self._handle_paper_detail_response(result, response_handler, session, is_followup=True)
                    else:
                        logger.error("[ROUTING] Failed to extract paper ID from most relevant paper")
                        await self._handle_error_response("Sorry, I couldn't find the paper details from our previous conversation.", response_handler)
                elif last_action == "paper_detail" and "current_paper" in session.metadata:
                    # Follow-up about the current paper
                    logger.info("[ROUTING] Following up on current paper")
                    paper = session.metadata["current_paper"]
                    await self._handle_paper_followup_response(paper, query.prompt, response_handler, session)
                else:
                    logger.error(f"[ROUTING] Cannot determine follow-up context. Last action: '{last_action}', Has last_papers: {('last_papers' in session.metadata)}")
                    # Can't determine what the follow-up is about
                    await self._handle_error_response("I'm not sure what you're asking about. Could you provide more details?", response_handler)
            
            elif query_type == QueryType.AGENT_INFO:
                logger.info("[ROUTING] Handling as AGENT_INFO query")
                result = await Tools.get_agent_info(query.prompt, session, self)
                await self._handle_agent_info_response(result, response_handler)
                
            elif query_type == QueryType.GREETING:
                logger.info("[ROUTING] Handling as GREETING query")
                await self._handle_greeting_response(query.prompt, response_handler)
                
            else:  # QueryType.UNKNOWN
                logger.info("[ROUTING] Handling as UNKNOWN query (defaulting to search)")
                # Try to handle as a search query by default
                result = await Tools.search_papers(query.prompt, session, self)
                await self._handle_search_response(result, response_handler, session)
                
            # Make sure to update the session store after handling the query
            await self._update_session(session)
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in assist: {str(e)}")
            logger.error(traceback.format_exc())
            await self._handle_error_response(f"I encountered an unexpected error. Please try again.", response_handler)
    
    def _log_session_state(self, session: Session):
        """Log the current session state for debugging."""
        if not hasattr(session, "metadata") or not session.metadata:
            logger.info("[SESSION] No session metadata available")
            return
            
        logger.info("[SESSION] Current session state:")
        for key, value in session.metadata.items():
            if key == "last_papers":
                logger.info(f"[SESSION] - last_papers: {len(value)} papers")
                for i, paper in enumerate(value[:2]):  # Log first 2 papers
                    logger.info(f"[SESSION]   - Paper {i+1}: '{paper.get('title')}' - ID: {paper.get('id')}")
            elif key == "current_paper":
                paper = value
                logger.info(f"[SESSION] - current_paper: '{paper.get('title')}' - ID: {paper.get('id')}")
            else:
                logger.info(f"[SESSION] - {key}: {value}")
    
    async def _classify_query(self, query_text: str, session: Session) -> Tuple[QueryType, Dict[str, Any]]:
        """
        Classify the query to determine its type and extract relevant entities.
        
        Args:
            query_text: The raw query text
            session: The session information
            
        Returns:
            A tuple of (QueryType, entities_dict)
        """
        logger.info(f"[CLASSIFY] Classifying query: '{query_text}'")
        
        # Convert to lowercase for easier matching
        text = query_text.lower()
        entities = {}
        
        # Check for greetings
        greeting_patterns = ["hello", "hi ", "hey", "greetings", "what's up"]
        if any(pattern in text.split() or text.startswith(pattern) for pattern in greeting_patterns):
            logger.info("[CLASSIFY] Matched greeting pattern")
            return QueryType.GREETING, {}
        
        # Check for agent info queries
        agent_info_patterns = ["who are you", "what can you do", "what are you", "help", "your capabilities"]
        if any(pattern in text for pattern in agent_info_patterns):
            logger.info(f"[CLASSIFY] Matched agent info pattern: '{next((p for p in agent_info_patterns if p in text), None)}'")
            return QueryType.AGENT_INFO, {}
        
        # Check for session context - do we have previous interactions?
        has_last_action = hasattr(session, "metadata") and "last_action" in session.metadata
        last_action = session.metadata.get("last_action") if has_last_action else None
        
        logger.info(f"[CLASSIFY] Follow-up check - Has last_action: {has_last_action}, Last action: '{last_action}'")
        
        # Enhanced check for follow-up queries
        # - Expanded patterns to catch more variations
        # - Added check for references to "this paper", "the paper", etc.
        followup_patterns = [
            "more details", "tell me more", "elaborate", "explain further", 
            "what about", "show me", "get me", "give me details", "dive deeper",
            "lets dive deeper", "can you dive deeper", "go deeper", 
            "this paper", "the paper", "that paper", "first paper",
            "details", "summary", "explain", "break down", "analyze", 
            "tell me about", "what does", "how does", "can i see"
        ]
        
        paper_reference_patterns = [
            "this paper", "the paper", "that paper", "the first paper", 
            "the top paper", "the most relevant paper"
        ]
        
        # First, check for explicit paper references if we have session context
        if has_last_action:
            explicit_paper_reference = any(pattern in text for pattern in paper_reference_patterns)
            if explicit_paper_reference:
                logger.info(f"[CLASSIFY] Matched explicit paper reference: {next((p for p in paper_reference_patterns if p in text), None)}")
                return QueryType.FOLLOWUP, {}
            
            # Then check for follow-up patterns
            if any(pattern in text for pattern in followup_patterns):
                matched_pattern = next((p for p in followup_patterns if p in text), None)
                logger.info(f"[CLASSIFY] Matched follow-up pattern: '{matched_pattern}'")
                return QueryType.FOLLOWUP, {}
            
            # Check for contextual follow-ups (short queries after previous results)
            is_short_query = len(text.split()) <= 5
            if is_short_query and (last_action == "search" or last_action == "paper_detail"):
                logger.info(f"[CLASSIFY] Short query ({len(text.split())} words) with previous context detected as follow-up")
                return QueryType.FOLLOWUP, {}
        
        # Check for arXiv ID patterns
        arxiv_id_pattern = r'(\d{4}\.\d{5})(v\d+)?'
        arxiv_url_pattern = r'arxiv\.org\/abs\/(\d{4}\.\d{5})(v\d+)?'
        
        arxiv_id_match = re.search(arxiv_id_pattern, query_text)
        arxiv_url_match = re.search(arxiv_url_pattern, query_text)
        
        if arxiv_id_match or arxiv_url_match:
            paper_id = arxiv_id_match.group(1) if arxiv_id_match else arxiv_url_match.group(1)
            entities["paper_id"] = paper_id
            logger.info(f"[CLASSIFY] Matched paper ID pattern: '{paper_id}'")
            return QueryType.PAPER_DETAIL, entities
        
        # Check for paper author patterns
        author_patterns = ["who is the author", "who wrote", "who are the authors", "authors of", "written by"]
        paper_indicators = ["paper", "article", "publication", "research"]
        
        if any(ap in text for ap in author_patterns) and any(pi in text for pi in paper_indicators):
            # This is an author question about a paper
            logger.info("[CLASSIFY] Matched author query pattern")
            
            # Extract the paper name/topic
            for ap in author_patterns:
                if ap in text:
                    rest = text.split(ap, 1)[1].strip()
                    for pi in paper_indicators:
                        if pi in rest:
                            paper_name = rest.split(pi, 1)[1].strip()
                            if paper_name:
                                # Remove "the", "of" and other common words
                                for word in ["the", "of", "about", "on", "for"]:
                                    paper_name = paper_name.replace(f" {word} ", " ").strip()
                                
                                entities["paper_topic"] = paper_name
                                logger.info(f"[CLASSIFY] Extracted paper topic: '{paper_name}'")
                                
                                if "last_papers" in session.metadata:
                                    # Try to find a matching paper in previous results
                                    last_papers = session.metadata["last_papers"]
                                    for paper in last_papers:
                                        title = paper.get("title", "").lower()
                                        if paper_name in title or any(word in title for word in paper_name.split() if len(word) > 3):
                                            logger.info(f"[CLASSIFY] Found matching paper in session: '{paper.get('title')}'")
                                            return QueryType.FOLLOWUP, {"paper_topic": paper_name}
            
            # If we couldn't find a match in session, treat it as a new search
            search_term = text
            entities["search_term"] = search_term
            logger.info(f"[CLASSIFY] Author query without session match, treating as search: '{search_term}'")
            return QueryType.SEARCH, entities
        
        # Check for search patterns
        search_patterns = ["find", "search", "look for", "papers about", "papers on", "research on", "articles about"]
        
        if any(pattern in text for pattern in search_patterns):
            # Extract the search term
            for pattern in search_patterns:
                if pattern in text:
                    search_term = text.split(pattern, 1)[1].strip()
                    if search_term:
                        entities["search_term"] = search_term
                        logger.info(f"[CLASSIFY] Matched search pattern: '{pattern}', Search term: '{search_term}'")
                        return QueryType.SEARCH, entities
        
        # Default: treat as search with the whole query as the search term
        entities["search_term"] = query_text
        logger.info(f"[CLASSIFY] Default classification as search query: '{query_text}'")
        return QueryType.SEARCH, entities
    
    def _extract_paper_id(self, paper: Dict[str, Any]) -> Optional[str]:
        """Extract a clean paper ID from a paper object."""
        if "id" not in paper:
            logger.error("[EXTRACT] Paper object has no 'id' field")
            return None
            
        paper_id_full = paper["id"]
        logger.info(f"[EXTRACT] Raw paper ID: '{paper_id_full}'")
        
        # Handle different ID formats
        if "/" in paper_id_full:
            paper_id = paper_id_full.split("/")[-1]
        else:
            paper_id = paper_id_full
            
        # Remove version if present
        if paper_id and "v" in paper_id:
            original_id = paper_id
            paper_id = paper_id.split("v")[0]
            logger.info(f"[EXTRACT] Removed version from ID: '{original_id}' -> '{paper_id}'")
            
        logger.info(f"[EXTRACT] Final extracted paper ID: '{paper_id}'")
        return paper_id
    
    async def _handle_search_response(
            self,
            result: Dict[str, Any],
            response_handler: ResponseHandler,
            session: Session
    ):
        """Handle the response for a search query."""
        logger.info("[RESPONSE] Handling search response")
        
        # Extract info from result
        query = result["query"]
        count = result["count"]
        search_results = result["results"]
        
        logger.info(f"[RESPONSE] Search results - Query: '{query}', Count: {count}")
        
        # Notify user about the search
        await response_handler.emit_text_block(
            "SEARCH", f"Searching arXiv for papers about {query}..."
        )
        
        # Emit search results as JSON
        if count > 0:
            await response_handler.emit_json(
                "PAPERS", {"results": search_results}
            )
            logger.info(f"[RESPONSE] Emitted {count} papers as JSON")
            
            try:
                # Create summary of search results
                logger.info("[RESPONSE] Generating search summary")
                summary_stream = response_handler.create_text_stream("SEARCH_SUMMARY")
                async for chunk in self._generate_search_summary(query, search_results):
                    filtered_chunk = self._filter_inappropriate_content(chunk)
                    await summary_stream.emit_chunk(filtered_chunk)
                await summary_stream.complete()
                logger.info("[RESPONSE] Search summary generation complete")
            except Exception as e:
                logger.error(f"[RESPONSE] Error generating search summary: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Create final response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            try:
                # Generate a concise, customized response that doesn't sound templated
                logger.info("[RESPONSE] Generating final response")
                prompt = f"""
                Generate a natural, conversational response about the papers I found related to "{query}".
                
                I found {count} papers. The most relevant is "{search_results[0]['title']}" by {', '.join(search_results[0]['authors'][:2])}.
                
                Make your response:
                1. Sound natural and conversational, not like a template
                2. Briefly mention the most interesting finding or contribution of the top paper
                3. Tell the user they can ask for more details about any paper
                4. Be concise (3-4 sentences total)
                5. Use formal academic language (absolutely no profanity or slang)
                6. Do not begin with "I found X papers related to Y" or any similar template phrases
                
                EXTREMELY IMPORTANT: DO NOT use informal language, slang or profanity of any kind. Keep your language strictly professional and academic.
                """
                
                # Custom system prompt for a more varied response with explicit content filtering
                system_prompt = (
                    "You are a knowledgeable research assistant who specializes in scientific literature. "
                    "Your responses are helpful, varied, and tailored to each query. "
                    "You always use professional, academic language without ANY profanity or slang. "
                    "You sound natural but scholarly and formal. "
                    "NEVER use inappropriate language, profanity, or casual slang terms. "
                    "Maintain the highest standards of academic discourse at all times."
                )
                
                logger.info(f"[RESPONSE] Calling model for final response with prompt of length {len(prompt)}")
                response = await self._model_provider.query(prompt, system_prompt)
                
                # Apply additional content filtering
                filtered_response = self._filter_inappropriate_content(response)
                
                logger.info(f"[RESPONSE] Generated final response of length {len(filtered_response)}")
                
                # Log a preview of the response
                preview = filtered_response[:100] + "..." if len(filtered_response) > 100 else filtered_response
                logger.info(f"[RESPONSE] Response preview: '{preview}'")
                
                await final_response_stream.emit_chunk(filtered_response)
            except Exception as e:
                logger.error(f"[RESPONSE] Error generating final response: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Fallback to a manual response
                most_relevant = search_results[0]
                fallback_response = f"""
                My search for "{query}" found {count} relevant papers. The most notable is "{most_relevant['title']}" by {', '.join(most_relevant['authors'][:2])}, which explores {most_relevant['summary'][:100]}...
                
                Would you like more details about this paper?
                """
                logger.info("[RESPONSE] Using fallback response")
                await final_response_stream.emit_chunk(fallback_response)
                
            await final_response_stream.complete()
        else:
            # No results found
            logger.info("[RESPONSE] No search results found")
            await response_handler.emit_text_block(
                "NO_RESULTS", f"No papers found matching your query on arXiv."
            )
            
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(
                f"I couldn't find any papers on arXiv matching your query about {query}. "
                f"Perhaps try a different search term or check if your query is too specific."
            )
            await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Search response handling complete")
    
    async def _handle_paper_detail_response(
            self,
            result: Dict[str, Any],
            response_handler: ResponseHandler,
            session: Session,
            is_followup: bool = False
    ):
        """Handle the response for a paper detail query."""
        logger.info(f"[RESPONSE] Handling paper detail response (is_followup: {is_followup})")
        
        paper_id = result["paper_id"]
        found = result["found"]
        
        # Notify user about fetching the paper
        await response_handler.emit_text_block(
            "FETCH_PAPER", f"Fetching details for paper {paper_id}..."
        )
        
        if found:
            paper = result["paper"]
            logger.info(f"[RESPONSE] Paper found: '{paper.get('title')}'")
            
            # Emit paper details as JSON
            await response_handler.emit_json(
                "PAPER_DETAILS", {"paper": paper}
            )
            
            try:
                # Generate and stream paper summary
                logger.info("[RESPONSE] Generating paper summary")
                summary_stream = response_handler.create_text_stream("PAPER_SUMMARY")
                async for chunk in self._model_provider.summarize_paper(paper, length="medium"):
                    filtered_chunk = self._filter_inappropriate_content(chunk)
                    await summary_stream.emit_chunk(filtered_chunk)
                await summary_stream.complete()
                logger.info("[RESPONSE] Paper summary generation complete")
            except Exception as e:
                logger.error(f"[RESPONSE] Error generating paper summary: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Create final response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # Extract paper details
            title = paper['title']
            authors = ', '.join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors += " et al."
            published = paper['published']
            categories = ', '.join(paper['categories'])
            summary = paper.get('summary', '').strip()
            
            logger.info(f"[RESPONSE] Preparing final response for paper: '{title}'")
            
            # Generate different response based on whether this is a follow-up
            if is_followup:
                # For follow-up queries, provide a more detailed response
                context = f"your search for \"{session.metadata.get('last_query', 'this topic')}\"" if "last_query" in session.metadata else "your previous query"
                logger.info(f"[RESPONSE] Using context: '{context}' for follow-up response")
                
                try:
                    prompt = f"""
                    Generate a natural, conversational response providing details about this paper that was found in {context}.
                    
                    Paper details:
                    Title: {title}
                    Authors: {authors}
                    Published: {published}
                    Categories: {categories}
                    
                    Abstract:
                    {summary}
                    
                    Make your response:
                    1. Sound natural and engaging, not like a template
                    2. Highlight 2-3 key points from the abstract
                    3. Mention the paper's significance to the field
                    4. Use formal academic language (no profanity or slang)
                    
                    EXTREMELY IMPORTANT: DO NOT use informal language, slang, or profanity of any kind. Keep your language strictly professional and academic.
                    DO NOT start with phrases like "Here are the details" or "Here are the detailed information".
                    """
                    
                    system_prompt = (
                        "You are a professional academic research assistant. "
                        "Your responses use formal scholarly language without ANY profanity or slang. "
                        "You sound natural but maintain proper academic tone at all times. "
                        "NEVER use inappropriate language, profanity, or casual slang terms. "
                        "Maintain the highest standards of academic discourse at all times."
                    )
                    
                    logger.info("[RESPONSE] Generating detailed follow-up response")
                    response = await self._model_provider.query(prompt, system_prompt)
                    
                    # Apply content filtering
                    filtered_response = self._filter_inappropriate_content(response)
                    
                    # Log a preview of the response
                    preview = filtered_response[:100] + "..." if len(filtered_response) > 100 else filtered_response
                    logger.info(f"[RESPONSE] Response preview: '{preview}'")
                    
                    # Add paper access links at the end
                    if "arxiv.org" not in filtered_response:
                        filtered_response += f"\n\nYou can access this paper at {paper['arxiv_url']} or download the PDF from {paper['pdf_url']}."
                    
                    await final_response_stream.emit_chunk(filtered_response)
                except Exception as e:
                    logger.error(f"[RESPONSE] Error generating detailed paper response: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Fallback response
                    fallback_response = f"""
                    The paper "{title}" by {authors} (published {published}) is in the {categories} field.
                    
                    Abstract:
                    {summary}
                    
                    You can access this paper at {paper['arxiv_url']} or download the PDF from {paper['pdf_url']}.
                    """
                    logger.info("[RESPONSE] Using fallback paper details response")
                    await final_response_stream.emit_chunk(fallback_response)
            else:
                # For direct paper requests, provide a more concise response
                try:
                    prompt = f"""
                    Generate a natural, conversational response about this paper the user requested by ID.
                    
                    Paper details:
                    Title: {title}
                    Authors: {authors}
                    Published: {published}
                    Categories: {categories}
                    
                    Abstract:
                    {summary}
                    
                    Make your response:
                    1. Brief introduction to the paper (1 sentence)
                    2. Summary of its main contribution (1-2 sentences)
                    3. Mention the paper's access links
                    4. Sound natural, not like a template
                    5. Use formal academic language (no profanity or slang)
                    
                    EXTREMELY IMPORTANT: DO NOT use informal language, slang or profanity of any kind. Keep your language strictly professional and academic.
                    DO NOT start with phrases like "I found the paper you requested".
                    """
                    
                    system_prompt = (
                        "You are a professional academic research assistant. "
                        "Your responses use formal scholarly language without ANY profanity or slang. "
                        "You sound natural but maintain proper academic tone at all times. "
                        "NEVER use inappropriate language, profanity, or casual slang terms. "
                        "Maintain the highest standards of academic discourse at all times."
                    )
                    
                    logger.info("[RESPONSE] Generating paper response")
                    response = await self._model_provider.query(prompt, system_prompt)
                    
                    # Apply content filtering
                    filtered_response = self._filter_inappropriate_content(response)
                    
                    # Log a preview of the response
                    preview = filtered_response[:100] + "..." if len(filtered_response) > 100 else filtered_response
                    logger.info(f"[RESPONSE] Response preview: '{preview}'")
                    
                    # Add paper access links if not included
                    if "arxiv.org" not in filtered_response:
                        filtered_response += f"\n\nYou can access this paper at {paper['arxiv_url']} or download the PDF from {paper['pdf_url']}."
                    
                    await final_response_stream.emit_chunk(filtered_response)
                except Exception as e:
                    logger.error(f"[RESPONSE] Error generating paper response: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Fallback response
                    fallback_response = f"""
                    The paper "{title}" by {authors} was published on {published} in the {categories} field.
                    
                    Abstract:
                    {summary[:300]}...
                    
                    You can access this paper at {paper['arxiv_url']} or download the PDF from {paper['pdf_url']}.
                    """
                    logger.info("[RESPONSE] Using fallback paper response")
                    await final_response_stream.emit_chunk(fallback_response)
            
            await final_response_stream.complete()
        else:
            # Paper not found
            error = result.get("error", "Unknown error")
            logger.error(f"[RESPONSE] Paper not found: {error}")
            
            await response_handler.emit_text_block(
                "ERROR", f"Could not find the paper: {error}"
            )
            
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(
                f"I couldn't find the paper with ID {paper_id} on arXiv. Please check if the ID is correct and try again."
            )
            await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Paper detail response handling complete")

    async def _handle_paper_followup_response(
            self,
            paper: Dict[str, Any],
            query_text: str,
            response_handler: ResponseHandler,
            session: Session
    ):
        """Handle follow-up questions about the current paper."""
        logger.info(f"[RESPONSE] Handling follow-up on current paper: '{paper.get('title')}'")
        
        # Extract paper details
        title = paper['title']
        authors = ', '.join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors += " et al."
        published = paper['published']
        categories = ', '.join(paper['categories'])
        summary = paper.get('summary', '').strip()
        
        # Emit paper details as JSON again for context
        await response_handler.emit_json(
            "PAPER_DETAILS", {"paper": paper}
        )
        
        # Create final response
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        try:
            # Generate a response focused on answering the specific follow-up question
            prompt = f"""
            Answer this follow-up question about a scientific paper:
            
            User's follow-up question: "{query_text}"
            
            Paper details:
            Title: {title}
            Authors: {authors}
            Published: {published}
            Categories: {categories}
            
            Abstract:
            {summary}
            
            Make your response:
            1. Directly address the user's specific question about the paper
            2. Use information from the paper details and abstract
            3. Sound natural and knowledgeable
            4. Use formal academic language (absolutely no profanity or slang)
            5. Be thorough but concise
            
            EXTREMELY IMPORTANT: DO NOT use informal language, slang or profanity of any kind. Keep your language strictly professional and academic.
            """
            
            system_prompt = (
                "You are a professional academic research assistant specializing in scientific papers. "
                "Your responses use formal scholarly language without ANY profanity or slang. "
                "You answer specific questions about academic papers with precision and nuance. "
                "NEVER use inappropriate language, profanity, or casual slang terms. "
                "Maintain the highest standards of academic discourse at all times."
            )
            
            logger.info("[RESPONSE] Generating paper follow-up response")
            response = await self._model_provider.query(prompt, system_prompt)
            
            # Apply content filtering
            filtered_response = self._filter_inappropriate_content(response)
            
            # Log a preview of the response
            preview = filtered_response[:100] + "..." if len(filtered_response) > 100 else filtered_response
            logger.info(f"[RESPONSE] Follow-up response preview: '{preview}'")
            
            await final_response_stream.emit_chunk(filtered_response)
        except Exception as e:
            logger.error(f"[RESPONSE] Error generating paper follow-up response: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback response
            fallback_response = f"""
            Regarding your question about "{title}":
            
            The paper was authored by {authors} and published on {published} in the {categories} field.
            
            Based on the abstract, the paper focuses on {summary[:150]}...
            
            You can access the full paper at {paper['arxiv_url']} for more details.
            """
            logger.info("[RESPONSE] Using fallback follow-up response")
            await final_response_stream.emit_chunk(fallback_response)
        
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Paper follow-up response handling complete")
    
    async def _handle_agent_info_response(
            self,
            result: Dict[str, Any],
            response_handler: ResponseHandler
    ):
        """Handle responses about the agent itself."""
        logger.info("[RESPONSE] Handling agent info response")
        
        name = result["name"]
        description = result["description"]
        capabilities = result["capabilities"]
        
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        # Create a concise, informative response about the agent
        response = f"""
I am {name}, an AI research assistant specialized in finding and explaining scientific papers from arXiv. 

I can help you:
- Search for papers on specific topics
- Retrieve and explain individual papers by ID
- Summarize research findings in accessible language
- Provide academic context for search results

Just ask me to find papers on a topic you're interested in, and I'll help you explore the research.
        """
        
        logger.info("[RESPONSE] Sending agent info response")
        await final_response_stream.emit_chunk(response)
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Agent info response complete")
    
    async def _handle_greeting_response(
            self,
            greeting: str,
            response_handler: ResponseHandler
    ):
        """Handle greeting responses."""
        logger.info("[RESPONSE] Handling greeting response")
        
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        responses = [
            "Hello! I'm your arXiv research assistant. What papers would you like me to find for you today?",
            "Hi there! I can help you search for scientific papers on arXiv. What topic are you interested in?",
            "Greetings! I'm here to help with your research. What scientific papers would you like to explore?",
            "Hello! Ready to dive into some research? Tell me what papers you're looking for."
        ]
        
        # Choose a random response for variety
        response = random.choice(responses)
        logger.info(f"[RESPONSE] Selected greeting response: '{response}'")
        
        await final_response_stream.emit_chunk(response)
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Greeting response complete")
    
    async def _handle_error_response(
            self,
            error_message: str,
            response_handler: ResponseHandler
    ):
        """Handle error responses."""
        logger.error(f"[RESPONSE] Error response: {error_message}")
        
        await response_handler.emit_text_block(
            "ERROR", error_message
        )
        
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        await final_response_stream.emit_chunk(
            f"{error_message} Please try again with a different query."
        )
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Error response complete")
    
    async def _generate_search_summary(
            self,
            query: str,
            papers: List[Dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Generate a summary of search results."""
        logger.info(f"[SUMMARY] Generating search summary for query: '{query}'")
        
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
        
        Use professional academic language throughout. ABSOLUTELY DO NOT use informal language, slang, or profanity.
        """
        
        # Custom system prompt for search summary with enhanced content filtering
        system_prompt = (
            "You are a professional academic research assistant specializing in scientific literature. "
            "You help researchers understand available papers on topics using formal, scholarly language. "
            "NEVER use casual expressions, slang, or profanity in your analyses. Maintain a professional, "
            "objective tone suitable for publication in an academic journal. "
            "Under no circumstances should you use ANY inappropriate language."
        )
        
        logger.info(f"[SUMMARY] Sending prompt of length {len(prompt)} to model")
        
        try:
            max_retries = 2
            retry_count = 0
            
            while True:
                try:
                    async for chunk in self._model_provider.query_stream(prompt, system_prompt):
                        filtered_chunk = self._filter_inappropriate_content(chunk)
                        yield filtered_chunk
                    logger.info("[SUMMARY] Successfully completed search summary generation")
                    break
                except (RateLimitError, APIError, APITimeoutError) as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"[SUMMARY] Max retries exceeded: {str(e)}")
                        raise
                    
                    delay = 2 ** retry_count * (1 + 0.1 * random.random())
                    logger.warning(f"[SUMMARY] API error in search summary: {str(e)}. Retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
        except Exception as e:
            # Provide a simpler fallback response
            logger.error(f"[SUMMARY] Error generating search summary: {str(e)}")
            logger.error(traceback.format_exc())
            
            logger.info("[SUMMARY] Using fallback summary response")
            simple_summary = f"Based on your query \"{query}\", I found {len(papers)} relevant papers:\n\n"
            yield simple_summary
            
            for i, paper in enumerate(papers[:3]):
                paper_summary = f"Paper {i+1}: \"{paper['title']}\" by {', '.join(paper['authors'][:2])}\n"
                paper_summary += f"Published: {paper['published']}\n"
                paper_summary += f"Summary: {paper['summary'][:150]}...\n\n"
                yield paper_summary
            
            yield "You can ask for more details about any of these papers."
            logger.info("[SUMMARY] Fallback summary generation complete")


if __name__ == "__main__":
    # Create an instance of the ArxivResearchAgent
    agent = ArxivResearchAgent(name="arXiv Research Agent")
    # Create an improved server to handle requests to the agent with proper session management
    server = ImprovedDefaultServer(agent)
    # Run the server
    server.run()