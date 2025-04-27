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
from datetime import datetime

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
    AUTHOR_PROFILE = "author"  # Get details about an author
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

# List of valid arXiv categories
VALID_ARXIV_CATEGORIES = [
    # Computer Science
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", 
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", 
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", 
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", 
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY",
    
    # Mathematics
    "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT", 
    "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GR", 
    "math.GT", "math.HO", "math.IT", "math.KT", "math.LO", "math.MG", "math.MP", 
    "math.NA", "math.NT", "math.OA", "math.OC", "math.PR", "math.QA", "math.RA", 
    "math.RT", "math.SG", "math.SP", "math.ST",
    
    # Physics
    "physics.acc-ph", "physics.ao-ph", "physics.atm-clus", "physics.atom-ph", 
    "physics.bio-ph", "physics.chem-ph", "physics.class-ph", "physics.comp-ph", 
    "physics.data-an", "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", 
    "physics.hist-ph", "physics.ins-det", "physics.med-ph", "physics.optics", 
    "physics.plasm-ph", "physics.pop-ph", "physics.soc-ph", "physics.space-ph",
    
    # Quantitative Biology
    "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO",
    
    # Quantitative Finance
    "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR",
    
    # Statistics
    "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"
]

# Define tools that the agent can use
class Tools:
    @staticmethod
    async def search_papers(
        query: str, 
        session: Session, 
        agent: 'ArxivResearchAgent',
        author: str = None,
        topic: str = None,
        date_range: str = None,
        sort_by: str = None,
        category: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced search for papers on arXiv with additional filters.
        
        Args:
            query: The basic search query text
            session: Session information
            agent: The ArxivResearchAgent instance
            author: Optional author name to filter results
            topic: Optional topic/category to filter results
            date_range: Optional date range for filtering (e.g., "last_month")
            sort_by: How to sort results - "relevance", "lastUpdatedDate", or "submittedDate"
            category: Optional arXiv category code to filter results (e.g., "cs.AI")
            
        Returns:
            Dictionary with search results including total count
        """
        logger.info(f"[TOOL] search_papers called with query: '{query}', author: '{author}', topic: '{topic}', sort_by: '{sort_by}', category: '{category}'")
        
        # Build search query with filters
        search_query = query.strip()
        search_params = []
        
        # Store the original user query for context
        original_query = query
        
        # Add author filter if provided
        if author:
            # Format author name for arXiv search
            formatted_author = f'au:"{author}"'
            search_params.append(formatted_author)
            logger.info(f"[TOOL] Added author filter: {formatted_author}")
        
        # Add category filter if provided
        if category:
            # Format category for arXiv search (ensuring it's a valid category code)
            if category in VALID_ARXIV_CATEGORIES:
                formatted_category = f'cat:{category}'
                search_params.append(formatted_category)
                logger.info(f"[TOOL] Added category filter: {formatted_category}")
            else:
                logger.warning(f"[TOOL] Invalid category code: {category}")
        
        # Add topic/category filter if provided (legacy support)
        if topic:
            # Format topic for arXiv search
            formatted_topic = f'cat:{topic}'
            search_params.append(formatted_topic)
            logger.info(f"[TOOL] Added topic filter: {formatted_topic}")
        
        # Add date range filter if provided
        if date_range:
            # Convert friendly date range to arXiv format
            if date_range == "last_day":
                search_params.append("submittedDate:[NOW-1DAY TO NOW]")
            elif date_range == "last_week":
                search_params.append("submittedDate:[NOW-7DAYS TO NOW]")
            elif date_range == "last_month":
                search_params.append("submittedDate:[NOW-1MONTH TO NOW]")
            elif date_range == "last_year":
                search_params.append("submittedDate:[NOW-1YEAR TO NOW]")
            # Support for custom date ranges (YYYY-MM-DD format)
            elif "to" in date_range.lower():
                try:
                    start_date, end_date = date_range.lower().split("to")
                    start_date = start_date.strip()
                    end_date = end_date.strip()
                    # Validate and format dates (implementation omitted)
                    search_params.append(f"submittedDate:[{start_date} TO {end_date}]")
                except Exception as e:
                    logger.error(f"[TOOL] Error parsing custom date range: {str(e)}")
            
            logger.info(f"[TOOL] Added date range filter: {date_range}")
        
        # Use the provided sort_by parameter if available
        if sort_by:
            logger.info(f"[TOOL] Using provided sort order: {sort_by}")
        else:
            # Default to most recent papers (submittedDate)
            sort_by = "submitteddate"
            logger.info(f"[TOOL] Using default sort order: {sort_by}")
        
        # Combine all search parameters
        if search_params:
            if search_query:
                # Combine with AND if there's already a query
                combined_query = f"{' AND '.join(search_params)} AND ({search_query})"
            else:
                # Just use the parameters if no query text
                combined_query = f"{' AND '.join(search_params)}"
            
            logger.info(f"[TOOL] Built combined query: {combined_query}")
        else:
            # Use the original query if no filters
            combined_query = search_query
        
        # Perform search
        try:
            search_response = await agent._arxiv_provider.search(
                combined_query, 
                max_results=5,
                sort_by=sort_by
            )
            search_results = search_response["results"]
            total_count = search_response["total_count"]
            
            logger.info(f"[TOOL] search_papers found {len(search_results)} results (total: {total_count})")
            
            # Log paper titles and IDs for debugging
            for i, paper in enumerate(search_results[:3]):
                logger.info(f"[TOOL] Paper {i+1}: '{paper.get('title')}' - ID: {paper.get('id')}")
            
            # Prepare result data
            result_data = {
                "type": "search_results",
                "query": original_query,
                "results": search_results,
                "total_count": total_count,
                "filters": {
                    "author": author,
                    "topic": topic,
                    "date_range": date_range,
                    "sort_by": sort_by,
                    "category": category
                }
            }
            
            # Update session state using the new management system
            await agent._manage_session_state(session, "search", result_data)
            
            return result_data
            
        except Exception as e:
            logger.error(f"[TOOL] Error in search_papers: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "type": "search_results",
                "query": original_query,
                "results": [],
                "total_count": 0,
                "filters": {
                    "author": author,
                    "topic": topic,
                    "date_range": date_range,
                    "sort_by": sort_by,
                    "category": category
                },
                "error": str(e)
            }
    
    @staticmethod
    async def get_author_profile(
        author_name: str, 
        session: Session, 
        agent: 'ArxivResearchAgent',
        max_papers: int = 10
    ) -> Dict[str, Any]:
        """
        Get detailed information about an author and their publication history.
        
        Args:
            author_name: Name of the author to look up
            session: Session information
            agent: The ArxivResearchAgent instance
            max_papers: Maximum number of papers to retrieve
            
        Returns:
            Dictionary with author profile and papers
        """
        logger.info(f"[TOOL] get_author_profile called with author: '{author_name}'")
        
        # Format the author query for arXiv
        author_query = f'au:"{author_name}"'
        logger.info(f"[TOOL] Author search query: {author_query}")
        
        try:
            # Search for papers by this author
            # Sort by submitted date to get the most recent papers first
            search_response = await agent._arxiv_provider.search(
                author_query, 
                max_results=max_papers,
                sort_by="submittedDate"
            )
            
            # Extract papers from response
            author_papers = search_response["results"]
            papers_found = len(author_papers)
            logger.info(f"[TOOL] Found {papers_found} papers by author '{author_name}'")
            
            # Extract categories to determine research interests
            categories = {}
            co_authors = set()
            years = {}
            
            # Process the papers to extract additional information
            for paper in author_papers:
                # Track categories/research areas
                for category in paper.get('categories', []):
                    categories[category] = categories.get(category, 0) + 1
                
                # Track co-authors
                for author in paper.get('authors', []):
                    if author != author_name:
                        co_authors.add(author)
                        
                # Track publication years
                year = paper.get('published', '').split('-')[0]
                if year:
                    years[year] = years.get(year, 0) + 1
            
            # Sort categories by frequency
            sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            top_categories = [{'category': cat, 'count': count} for cat, count in sorted_categories[:5]]
            
            # Sort years chronologically
            sorted_years = sorted(years.items())
            publication_timeline = [{'year': year, 'papers': count} for year, count in sorted_years]
            
            # Build the author profile
            profile = {
                "name": author_name,
                "paper_count": papers_found,
                "papers": author_papers,
                "research_interests": top_categories,
                "co_authors": list(co_authors)[:10],  # Limit to top 10 co-authors
                "publication_timeline": publication_timeline
            }
            
            # Prepare result data
            result_data = {
                "type": "author_profile",
                "author": author_name,
                "found": True,
                "profile": profile,
                "papers": author_papers
            }
            
            # Update session state using the new management system
            await agent._manage_session_state(session, "author_profile", result_data)
            
            return result_data
            
        except Exception as e:
            logger.error(f"[TOOL] Error fetching author profile for {author_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                "type": "author_profile",
                "author": author_name,
                "found": False,
                "error": str(e)
            }
    
    @staticmethod
    async def get_paper_details(paper_id: str, session: Session, agent: 'ArxivResearchAgent') -> Dict[str, Any]:
        """Get detailed information about a specific paper."""
        logger.info(f"[TOOL] get_paper_details called with paper_id: '{paper_id}'")
        
        # Get paper details
        try:
            # Extract a more specific part of paper_id to avoid errors
            # (This function is now improved to handle more ID formats)
            paper = await agent._arxiv_provider.get_paper_by_id(paper_id)
            logger.info(f"[TOOL] Successfully retrieved paper: '{paper.get('title')}'")
            
            # Prepare result data
            result_data = {
                "type": "paper_detail",
                "paper_id": paper_id,
                "paper": paper,
                "found": True
            }
            
            # Update session state using the new management system
            await agent._manage_session_state(session, "paper_detail", result_data)
            
            return result_data
            
        except Exception as e:
            logger.error(f"[TOOL] Error fetching paper {paper_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to recover - if we have the paper in last_papers, use that instead
            if hasattr(session, "metadata") and "last_papers" in session.metadata:
                for paper in session.metadata["last_papers"]:
                    if paper_id in paper.get('id', ''):
                        logger.info(f"[TOOL] Recovered paper from session: '{paper.get('title')}'")
                        
                        result_data = {
                            "type": "paper_detail",
                            "paper_id": paper_id,
                            "paper": paper,
                            "found": True
                        }
                        
                        # Update session state using the new management system
                        await agent._manage_session_state(session, "paper_detail", result_data)
                        
                        return result_data
            
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
                "Find papers by specific authors",
                "Generate summaries of research findings",
                "Provide academic context for search results"
            ]
        }


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
        # Ensure session metadata exists
        if not hasattr(session, 'metadata') or not session.metadata:
            session.metadata = {}
            logger.info("[SESSION] Initialized empty session metadata in _update_session")
        
        # Get activity_id from session metadata or context
        activity_id = None
        if hasattr(session, '_session_object'):
            # Try to get activity_id from session object
            activity_id = getattr(session._session_object, 'activity_id', None)
            if activity_id:
                session.metadata['activity_id'] = str(activity_id)
                logger.info(f"[SESSION] Set activity_id from session object: {activity_id}")
        
        # If still no activity_id, try to get it from metadata
        if not activity_id:
            activity_id = session.metadata.get('activity_id')
            if not activity_id:
                logger.warning("[SESSION] No activity_id found in session metadata")
                return
        
        # Store session metadata using the activity ID
        self._session_store[activity_id] = session.metadata
        logger.info(f"[SESSION] Updated persistent session store for activity {activity_id}")

    async def _manage_session_state(self, session: Session, action: str, data: Dict[str, Any] = None) -> None:
        """Manage session state based on actions."""
        if not hasattr(session, 'metadata') or not session.metadata:
            session.metadata = {'activity_id': 'unknown'}
            logger.info("[SESSION] Initialized session metadata in _manage_session_state")
        
        # Ensure activity_id is set in metadata
        if 'activity_id' not in session.metadata:
            session.metadata['activity_id'] = 'unknown'
        
        if action == "search":
            if data and 'query' in data:
                session.metadata['last_search_query'] = data['query']
                session.metadata['last_search_time'] = datetime.now().isoformat()
        elif action == "paper_detail":
            if data and 'paper_id' in data:
                session.metadata['last_viewed_paper'] = data['paper_id']
                session.metadata['last_viewed_time'] = datetime.now().isoformat()
        elif action == "author_profile":
            if data and 'author_id' in data:
                session.metadata['last_viewed_author'] = data['author_id']
                session.metadata['last_viewed_time'] = datetime.now().isoformat()
        
        # Update session in persistent store
        await self._update_session(session)

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
        """Handle incoming requests and manage session state."""
        try:
            # Log raw request data
            print("\n" + "=" * 80)
            print("[ASSIST] RAW REQUEST DATA:")
            print(f"Session: {json.dumps(session.__dict__, indent=2, default=str)}")
            print(f"Query: {json.dumps(query.__dict__, indent=2, default=str)}")
            print("=" * 80 + "\n")
            
            # Classify the query
            query_type, query_data = await self._classify_query(query.prompt, session)
            
            # Handle the query based on its type
            if query_type == QueryType.SEARCH:
                result = await Tools.search_papers(
                    query=query.prompt,
                    session=session,
                    agent=self,
                    author=query_data.get('author'),
                    topic=query_data.get('topic'),
                    date_range=query_data.get('date_range'),
                    sort_by=query_data.get('sort_by'),
                    category=query_data.get('category')
                )
                await self._handle_search_response(result, response_handler, session)
                
            elif query_type == QueryType.PAPER_DETAIL:
                paper_id = query_data.get('paper_id')
                if not paper_id:
                    await self._handle_error_response(
                        "Please provide a paper ID or reference a paper from previous results.",
                        response_handler
                    )
                    return
                
                result = await Tools.get_paper_details(paper_id, session, self)
                await self._handle_paper_detail_response(result, response_handler, session)
                
            elif query_type == QueryType.AUTHOR_PROFILE:
                author_name = query_data.get('author_name')
                if not author_name:
                    await self._handle_error_response(
                        "Please provide an author name to search for.",
                        response_handler
                    )
                    return
                
                result = await Tools.get_author_profile(author_name, session, self)
                await self._handle_author_profile_response(result, response_handler, session)
                
            elif query_type == QueryType.FOLLOWUP:
                if not session.metadata.get('last_papers'):
                    await self._handle_error_response(
                        "No previous search results found. Please perform a search first.",
                        response_handler
                    )
                    return
                
                await self._handle_paper_followup_response(
                    session.metadata['last_papers'][0],
                    query.prompt,
                    response_handler,
                    session
                )
                
            elif query_type == QueryType.AGENT_INFO:
                result = await Tools.get_agent_info(query.prompt, session, self)
                await self._handle_agent_info_response(result, response_handler)
                
            elif query_type == QueryType.GREETING:
                await self._handle_greeting_response(query.prompt, response_handler)
                
            else:
                await self._handle_error_response(
                    "I couldn't understand your request. Please try rephrasing it.",
                    response_handler
                )
            
            # Update session state
            await self._update_session(session)
            
            # Mark response as complete
            await response_handler.complete()
            
        except Exception as e:
            await self._handle_error_response(str(e), response_handler)
    
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

                
    
    async def _classify_query(
            self,
            query_text: str,
            session: Session
    ) -> Tuple[QueryType, Dict[str, Any]]:
        """
        Classify the query using LLM first, falling back to pattern matching.
        
        Args:
            query_text: The raw query text
            session: The session information
                
        Returns:
            A tuple of (QueryType, entities_dict)
        """
        try:
            # Try LLM classification first
            return await self._classify_query_with_llm(query_text, session)
        except Exception as e:
            logger.error(f"[CLASSIFY] LLM classification failed, falling back to pattern-based: {str(e)}")
            return await self._classify_query_pattern_based(query_text, session)

    async def _classify_query_with_llm(self, query_text: str, session: Session) -> Tuple[QueryType, Dict[str, Any]]:
        """
        Use the LLM to classify the query and extract relevant entities.
        
        Args:
            query_text: The raw query text
            session: The session information
                
        Returns:
            A tuple of (QueryType, entities_dict)
        """
        logger.info(f"[CLASSIFY] Classifying query with LLM: '{query_text}'")
        
        # Build context from session
        context = "No previous conversation context."
        if hasattr(session, "metadata"):
            if "last_action" in session.metadata:
                context = f"The last action was: {session.metadata['last_action']}"
                
                if session.metadata["last_action"] == "search" and "last_papers" in session.metadata:
                    papers = session.metadata["last_papers"]
                    paper_titles = [p.get('title', 'Unknown') for p in papers[:3]]
                    context += f"\nThe last search returned these papers: {', '.join(paper_titles)}"
                    if "last_query" in session.metadata:
                        context += f"\nThe search query was: {session.metadata['last_query']}"
                        
                elif session.metadata["last_action"] == "paper_detail" and "current_paper" in session.metadata:
                    paper = session.metadata["current_paper"]
                    context += f"\nThe current paper being discussed is: {paper.get('title', 'Unknown')}"
        
        # Create prompt for classification
        system_prompt = """You are a query analyzer for an arXiv research assistant. Classify the user's query type and extract entities.

Query types:
- SEARCH: User wants to search for papers on a topic
- PAPER_DETAIL: User wants details about a specific paper
- AUTHOR_PROFILE: User wants information about an author's work and background
- FOLLOWUP: User is asking a follow-up about previous results
- AGENT_INFO: User is asking about the agent itself
- GREETING: A simple greeting
- UNKNOWN: Cannot classify

Output JSON format:
{
  "query_type": "TYPE",
  "entities": {
    "search_term": "extracted search term",
    "paper_id": "extracted paper ID",
    "author": "extracted author name",
    "topic": "extracted topic/category",
    "date_range": "extracted date range",
    "sort_by": "how to sort (relevance, submittedDate, lastUpdatedDate)",
    "list_all_papers": true/false
  }
}

IMPORTANT RULES:
1. If the query explicitly asks about an author (e.g., "tell me about author X", "show me papers by X", "who is X") classify as AUTHOR_PROFILE.
2. If the query mentions a specific paper by ID or title, classify as PAPER_DETAIL.
3. For AUTHOR_PROFILE, always include the "author" entity with the full author name.
4. Only include entities that are present in the query.
5. For SEARCH, include sort_by="submittedDate" if no specific query is present, as users typically want recent papers by default.
"""
        
        prompt = f"""
        Classify this query from a user talking to an arXiv research assistant:
        
        USER QUERY: {query_text}
        
        CONVERSATION CONTEXT:
        {context}
        
        Based on the query and context, determine the query type and extract relevant entities.
        Return ONLY a valid JSON object with "query_type" and "entities" fields.
        """
        
        try:
            # Call the model for classification
            response = await self._model_provider.query(prompt, system_prompt)
            
            # Clean the response to extract just the JSON part
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
            else:
                json_str = response
                
            # Parse the JSON response
            try:
                classification = json.loads(json_str)
                
                # Get query type
                query_type_str = classification.get("query_type", "UNKNOWN").upper()
                try:
                    query_type = QueryType[query_type_str]
                except KeyError:
                    logger.error(f"[CLASSIFY] Invalid query type returned: {query_type_str}")
                    query_type = QueryType.UNKNOWN
                    
                # Get entities
                entities = classification.get("entities", {})
                
                # If we have a paper_title, try to find the matching paper in last_papers
                if "paper_title" in entities and "last_papers" in session.metadata:
                    paper_title = entities["paper_title"].lower()
                    for paper in session.metadata["last_papers"]:
                        if paper_title in paper.get("title", "").lower():
                            entities["paper_id"] = self._extract_paper_id(paper)
                            logger.info(f"[CLASSIFY] Found matching paper for title '{paper_title}': {entities['paper_id']}")
                            break
                
                # Normalize sort_by and date_range values if present
                if "sort_by" in entities:
                    sort_value = entities["sort_by"].lower()
                    if sort_value in ["relevance", "lastupdateddate", "submitteddate"]:
                        entities["sort_by"] = sort_value
                    else:
                        logger.warning(f"[CLASSIFY] Invalid sort_by value: {sort_value}")
                        del entities["sort_by"]
                
                if "date_range" in entities:
                    date_value = entities["date_range"].lower()
                    valid_ranges = ["last_day", "last_week", "last_month", "last_year"]
                    if date_value in valid_ranges or "to" in date_value:
                        entities["date_range"] = date_value
                    else:
                        logger.warning(f"[CLASSIFY] Invalid date_range value: {date_value}")
                        del entities["date_range"]
                
                logger.info(f"[CLASSIFY] LLM classified as: {query_type.value}, Entities: {entities}")
                return query_type, entities
                
            except json.JSONDecodeError as e:
                logger.error(f"[CLASSIFY] Failed to parse JSON from LLM: {response}")
                logger.error(f"[CLASSIFY] JSON error: {str(e)}")
                # Only fall back if we can't parse the JSON at all
                return await self._classify_query_pattern_based(query_text, session)
                
        except Exception as e:
            logger.error(f"[CLASSIFY] Error in LLM classification: {str(e)}")
            logger.error(traceback.format_exc())
            # Only fall back if there's a complete failure
            return await self._classify_query_pattern_based(query_text, session)

    async def _classify_query_pattern_based(self, query_text: str, session: Session) -> Tuple[QueryType, Dict[str, Any]]:
        """
        Pattern-based classification of queries to determine intent and extract parameters.
        
        Args:
            query_text: The raw query text
            session: The session information
                
        Returns:
            A tuple of (QueryType, entities_dict)
        """
        logger.info(f"[CLASSIFY] Using pattern-based classification for query: '{query_text}'")
        
        # Convert to lowercase for easier matching
        text = query_text.lower()
        entities = {}
        
        # Check for greetings
        greeting_patterns = ["hello", "hi ", "hey", "greetings", "what's up"]
        for pattern in greeting_patterns:
            if text.startswith(pattern) or pattern.strip() == text:
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
        
        # Check for "list all papers" type queries - These should be treated as follow-ups
        # when we have previous search results
        list_papers_patterns = [
            "list all papers", "show all papers", "what papers did you find", 
            "what other papers", "show me all papers", "what else did you find",
            "list the papers", "show papers", "all papers", "the other papers"
        ]
        
        if has_last_action and last_action == "search" and "last_papers" in session.metadata:
            if any(pattern in text for pattern in list_papers_patterns):
                matched_pattern = next((p for p in list_papers_patterns if p in text), None)
                logger.info(f"[CLASSIFY] Matched list papers pattern: '{matched_pattern}'")
                return QueryType.FOLLOWUP, {"list_all_papers": True}
        
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
        
        # Author search patterns - NEW
        author_search_patterns = [
            "papers by this author", "more papers by this author", 
            "other papers by this author", "find papers by this author",
            "author's other papers", "more from this author", 
            "what else has this author written", "other research by this author",
            "same author", "other works by"
        ]
        
        # Check for author search patterns if we have a current paper
        if has_last_action and "current_paper" in session.metadata:
            current_paper = session.metadata["current_paper"]
            if any(pattern in text for pattern in author_search_patterns):
                # Extract author from current paper
                if "authors" in current_paper and current_paper["authors"]:
                    author = current_paper["authors"][0]  # First author
                    entities["author"] = author
                    entities["search_term"] = ""  # Empty search term, just use author filter
                    logger.info(f"[CLASSIFY] Detected author search for: '{author}'")
                    return QueryType.SEARCH, entities
        
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
        old_arxiv_pattern = r'(\w+)\/(\d{7})(v\d+)?'  # e.g., physics/0609047
        
        arxiv_id_match = re.search(arxiv_id_pattern, query_text)
        arxiv_url_match = re.search(arxiv_url_pattern, query_text)
        old_arxiv_match = re.search(old_arxiv_pattern, query_text)
        
        if arxiv_id_match:
            paper_id = arxiv_id_match.group(1)
            entities["paper_id"] = paper_id
            logger.info(f"[CLASSIFY] Matched paper ID pattern: '{paper_id}'")
            return QueryType.PAPER_DETAIL, entities
        elif arxiv_url_match:
            paper_id = arxiv_url_match.group(1)
            entities["paper_id"] = paper_id
            logger.info(f"[CLASSIFY] Matched paper URL pattern: '{paper_id}'")
            return QueryType.PAPER_DETAIL, entities
        elif old_arxiv_match:
            category = old_arxiv_match.group(1)
            id_num = old_arxiv_match.group(2)
            paper_id = f"{category}/{id_num}"
            entities["paper_id"] = paper_id
            logger.info(f"[CLASSIFY] Matched old-style paper ID pattern: '{paper_id}'")
            return QueryType.PAPER_DETAIL, entities
        
        # Check for explicit author search patterns
        author_patterns = ["papers by author", "author search", "find author", "search author"]
        
        for pattern in author_patterns:
            if pattern in text:
                # Extract author name after the pattern
                parts = text.split(pattern, 1)
                if len(parts) > 1 and parts[1].strip():
                    author_name = parts[1].strip()
                    # Clean up author name - remove leading "by" if present
                    if author_name.startswith("by "):
                        author_name = author_name[3:].strip()
                    
                    entities["author"] = author_name
                    entities["search_term"] = ""  # Empty search term, just use author filter
                    logger.info(f"[CLASSIFY] Explicit author search for: '{author_name}'")
                    return QueryType.SEARCH, entities
        
        # Check for paper author patterns (questions about authors)
        author_question_patterns = ["who is the author", "who wrote", "who are the authors", "authors of", "written by"]
        paper_indicators = ["paper", "article", "publication", "research"]
        
        if any(ap in text for ap in author_question_patterns) and any(pi in text for pi in paper_indicators):
            # This is an author question about a paper
            logger.info("[CLASSIFY] Matched author query pattern")
            
            # Extract the paper name/topic
            for ap in author_question_patterns:
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
        
        # Handle different ID formats for arXiv
        # Format could be:
        # - http://arxiv.org/abs/physics/0609047v1 (old format with category)
        # - http://arxiv.org/abs/2407.20030v1 (new format)
        # - direct IDs like "2407.20030v1" or "physics/0609047v1"
        
        # First extract the ID part from the URL if needed
        if "arxiv.org" in paper_id_full:
            # Extract ID from URL
            parts = paper_id_full.split("/abs/")
            if len(parts) > 1:
                paper_id = parts[1]
            else:
                # Try alternate format
                parts = paper_id_full.split("/")
                paper_id = parts[-1]
        else:
            paper_id = paper_id_full
        
        # For old-style IDs (with category prefix like "physics/0609047v1")
        # Format correctly for arXiv API
        if "/" in paper_id and not paper_id.startswith("http"):
            # This is already in the correct format for arXiv API, just remove version if present
            if "v" in paper_id.split("/")[1]:
                category_parts = paper_id.split("/")
                id_part = category_parts[1]
                version_parts = id_part.split("v")
                clean_id = f"{category_parts[0]}/{version_parts[0]}"
                logger.info(f"[EXTRACT] Formatted old-style ID with category: '{paper_id}' -> '{clean_id}'")
                return clean_id
            return paper_id  # Already in correct format
            
        # For new-style IDs (just numbers like "2407.20030v1")
        # Handle version numbers if present
        if "v" in paper_id and not paper_id.startswith("http"):
            version_parts = paper_id.split("v")
            clean_id = version_parts[0]  # Remove version number
            logger.info(f"[EXTRACT] Removed version from ID: '{paper_id}' -> '{clean_id}'")
            paper_id = clean_id
        
        logger.info(f"[EXTRACT] Final extracted paper ID: '{paper_id}'")
        return paper_id
    
    async def _handle_search_response(
            self,
            result: Dict[str, Any],
            response_handler: ResponseHandler,
            session: Session
    ):
        """Enhanced handler for search responses that provides an overview of all papers."""
        logger.info("[RESPONSE] Handling search response")
        
        # Extract info from result
        query = result["query"]
        count = result["count"]
        search_results = result["results"]
        filters = result.get("filters", {})
        
        # Extract filter information for the response
        author_filter = filters.get("author")
        topic_filter = filters.get("topic")
        date_filter = filters.get("date_range")
        
        logger.info(f"[RESPONSE] Search results - Query: '{query}', Count: {count}, Filters: {filters}")
        
        # Create notification message that includes filter information
        search_message = f"Searching arXiv for papers"
        if query:
            search_message += f" about {query}"
        if author_filter:
            search_message += f" by author {author_filter}"
        if topic_filter:
            search_message += f" in category {topic_filter}"
        if date_filter:
            search_message += f" from {date_filter}"
        search_message += "..."
        
        # Notify user about the search
        await response_handler.emit_text_block(
            "SEARCH", search_message
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
                async for chunk in self._generate_search_summary(query, search_results, author=author_filter):
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
                # Generate a concise, customized response that includes ALL papers
                logger.info("[RESPONSE] Generating final response with all papers")
                
                # Build a prompt that requests summaries of all papers
                prompt = f"""
                Generate a natural, conversational response about the {count} papers I found related to "{query}".
                
                Here are the papers:
                {json.dumps([{
                    "title": paper.get("title"),
                    "authors": ", ".join(paper.get("authors", [])[:2]) + (" et al." if len(paper.get("authors", [])) > 2 else ""),
                    "published": paper.get("published"),
                    "primary_category": paper.get("primary_category", ""),
                    "summary_snippet": paper.get("summary", "")[:150] + "..."
                } for paper in search_results], indent=2)}
                
                Make your response:
                1. Include a brief mention of EACH paper with its title and main topic
                2. Group papers by theme if possible
                3. Do NOT focus only on the first/most relevant paper
                4. Tell the user they can ask for more details about any specific paper
                5. Be concise but comprehensive
                6. Use formal academic language (no profanity or slang)
                
                EXTREMELY IMPORTANT: 
                - DO NOT use informal language or slang
                - DO NOT say "I found X papers" - instead, describe the papers themselves
                - MENTION EACH PAPER by title (in quotes) and briefly what it's about
                - Include the PAPER NUMBER (1-5) before each title to make it easy for the user to reference
                """
                
                # Custom system prompt for a comprehensive response
                system_prompt = (
                    "You are a knowledgeable research assistant who specializes in scientific literature. "
                    "Your responses are helpful, varied, and tailored to each query. "
                    "When presenting search results, you briefly describe EACH paper found, "
                    "not just the most relevant one. "
                    "You maintain a scholarly tone while being conversational and helpful."
                )
                
                logger.info(f"[RESPONSE] Calling model for final response with prompt of length {len(prompt)}")
                response = await self._model_provider.query(prompt, system_prompt)
                
                # Apply content filtering
                filtered_response = self._filter_inappropriate_content(response)
                
                logger.info(f"[RESPONSE] Generated final response of length {len(filtered_response)}")
                
                # Log a preview of the response
                preview = filtered_response[:100] + "..." if len(filtered_response) > 100 else filtered_response
                logger.info(f"[RESPONSE] Response preview: '{preview}'")
                
                await final_response_stream.emit_chunk(filtered_response)
            except Exception as e:
                logger.error(f"[RESPONSE] Error generating final response: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Fallback response that still covers all papers
                fallback_response = f"Here are the papers I found related to \"{query}\":\n\n"
                
                for i, paper in enumerate(search_results):
                    fallback_response += f"{i+1}. \"{paper.get('title')}\" by {', '.join(paper.get('authors', [])[:2])}"
                    if len(paper.get('authors', [])) > 2:
                        fallback_response += " et al."
                    fallback_response += f" ({paper.get('published')})\n"
                    fallback_response += f"   Category: {paper.get('primary_category', 'N/A')}\n"
                    
                    # Add a brief snippet of the summary
                    summary = paper.get('summary', '')
                    if summary:
                        summary_snippet = summary[:150] + "..." if len(summary) > 150 else summary
                        fallback_response += f"   Summary: {summary_snippet}\n"
                    
                    fallback_response += "\n"
                
                fallback_response += "You can ask for more details about any of these papers by referring to its number or title."
                
                logger.info("[RESPONSE] Using fallback response with all papers")
                await final_response_stream.emit_chunk(fallback_response)
                
            await final_response_stream.complete()
        else:
            # No results found
            logger.info("[RESPONSE] No search results found")
            
            # Create error message that includes filter context
            no_results_message = "No papers found"
            if author_filter:
                no_results_message += f" by author {author_filter}"
            if query:
                no_results_message += f" matching your query about {query}"
            no_results_message += " on arXiv."
            
            await response_handler.emit_text_block(
                "NO_RESULTS", no_results_message
            )
            
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # Build suggestion based on filters
            suggestion = ""
            if author_filter:
                suggestion = "Try checking the spelling of the author name or searching for a different author."
            else:
                suggestion = "Perhaps try a different search term or check if your query is too specific."
            
            await final_response_stream.emit_chunk(
                f"I couldn't find any papers{' by ' + author_filter if author_filter else ''}{' about ' + query if query else ''} on arXiv. {suggestion}"
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
                    5. Mention that the user can ask to find more papers by the same author
                    
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
                    
                    If you'd like to see more papers by {paper['authors'][0]}, just ask.
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
                    6. Mention that the user can ask to find more papers by the first author
                    
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
                    
                    If you'd like to see more papers by {paper['authors'][0]}, just ask.
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
            
            # Create a more helpful error response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # Check if we have previous search results to recommend
            if "last_papers" in session.metadata and len(session.metadata["last_papers"]) > 0:
                papers = session.metadata["last_papers"]
                fallback_response = f"""
                I couldn't find the specific paper with ID {paper_id}. This might be due to an error in the ID format or the paper may not be available on arXiv.
                
                However, based on your previous search, I can tell you about these papers instead:
                
                "{papers[0]['title']}" by {', '.join(papers[0]['authors'][:2])}
                "{papers[1]['title']}" by {', '.join(papers[1]['authors'][:2])}
                
                Would you like me to get details about one of these papers instead?
                """
                logger.info("[RESPONSE] Using fallback with suggested papers")
                await final_response_stream.emit_chunk(fallback_response)
            else:
                # Basic error if no previous results
                await final_response_stream.emit_chunk(
                    f"I couldn't find the paper with ID {paper_id} on arXiv. This might be due to an error in the ID format or the paper might not be accessible. Please check if the ID is correct and try again, or try searching for the paper by its title."
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
            6. If relevant, mention that the user can ask to find more papers by the first author
            
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
            
            If you'd like to see more papers by {paper['authors'][0]}, feel free to ask.
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
        
        # Create a prompt for the LLM to generate a more natural response about the agent
        try:
            prompt = f"""
            Generate a natural, conversational response explaining who you are as {name}.
            
            About you:
            {description}
            
            Your capabilities:
            {', '.join(capabilities)}
            
            Make your response:
            1. Sound natural and conversational
            2. Be informative about what you can help with
            3. Invite the user to ask about papers they're interested in
            4. Be friendly but professional
            5. Do not use bullet points or lists
            """
            
            system_prompt = (
                "You are a helpful research assistant specializing in scientific papers. "
                "Your responses are natural, helpful, and inviting. "
                "You explain your capabilities clearly but conversationally."
            )
            
            logger.info("[RESPONSE] Generating agent info response with LLM")
            response = await self._model_provider.query(prompt, system_prompt)
            
            # Create final response stream
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(response)
            await final_response_stream.complete()
            
        except Exception as e:
            logger.error(f"[RESPONSE] Error generating LLM agent info: {str(e)}")
            
            # Fall back to template response if LLM fails
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # Create a concise, informative response about the agent
            fallback_response = f"""
            I am {name}, an AI research assistant specialized in finding and explaining scientific papers from arXiv. 

            I can help you search for papers on specific topics, retrieve and explain individual papers by ID, find papers by specific authors, summarize research findings, and provide academic context for search results.

            Just ask me to find papers on a topic you're interested in, and I'll help you explore the research.
            """
            
            logger.info("[RESPONSE] Using fallback agent info response")
            await final_response_stream.emit_chunk(fallback_response)
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
        
        # Try to use the LLM for a more varied greeting
        try:
            prompt = """
            Generate a friendly, brief greeting as an arXiv research agent. 
            Introduce yourself as an AI agent that can help users find and understand scientific papers from arXiv.
            Explain that you can search for papers, provide details about specific papers, help explore research topics, and look up author profiles and their publication history.
            Keep it to 2-3 sentences, sound natural and varied (not templated).
            """
            
            system_prompt = (
                "You are an AI research agent specializing in finding and explaining scientific papers from arXiv. "
                "Your responses are brief, friendly, and conversational while maintaining a professional tone."
            )
            
            logger.info("[RESPONSE] Generating LLM greeting response")
            response = await self._model_provider.query(prompt, system_prompt)
            
            # Create final response stream
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(response)
            await final_response_stream.complete()
            
        except Exception as e:
            logger.error(f"[RESPONSE] Error generating LLM greeting: {str(e)}")
            
            # Fall back to template responses if LLM fails
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            responses = [
                "Hello! I'm your arXiv research assistant. What papers would you like me to find for you today?",
                "Hi there! I can help you search for scientific papers on arXiv. What topic are you interested in?",
                "Greetings! I'm here to help with your research. What scientific papers would you like to explore?",
                "Hello! Ready to dive into some research? Tell me what papers you're looking for."
            ]
            
            # Choose a random response for variety
            response = random.choice(responses)
            logger.info(f"[RESPONSE] Selected fallback greeting response: '{response}'")
            
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
    
    async def _handle_list_all_papers_response(
            self,
            session: Session,
            response_handler: ResponseHandler
    ):
        """
        Handle requests to list all papers from previous searches.
        This combines papers from multiple searches if available.
        """
        logger.info("[RESPONSE] Handling list all papers request")
        
        # Get the most recent search results
        last_papers = session.metadata.get("last_papers", [])
        previous_search_query = session.metadata.get("last_query", "your search")
        
        if not last_papers:
            await response_handler.emit_text_block(
                "NO_RESULTS", "No papers found in your previous searches."
            )
            
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await final_response_stream.emit_chunk(
                "I don't have any previous search results to show. Please try searching for papers first."
            )
            await final_response_stream.complete()
            await response_handler.complete()
            return
        
        # Emit the papers as JSON
        await response_handler.emit_json(
            "PAPERS", {"results": last_papers}
        )
        logger.info(f"[RESPONSE] Emitted {len(last_papers)} papers from previous search")
        
        # Create final response
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        try:
            # Generate a response listing all papers with brief descriptions
            prompt = f"""
            Create a concise, formatted list of the papers from the previous search on "{previous_search_query}".
            
            Here are the papers:
            {json.dumps([{
                "title": paper.get("title"),
                "authors": ", ".join(paper.get("authors", [])[:2]),
                "published": paper.get("published"),
                "summary": paper.get("summary", "")[:100] + "..."
            } for paper in last_papers], indent=2)}
            
            For each paper:
            1. Start with the title in quotes, followed by the authors
            2. Add a very brief (1 sentence) description of what the paper is about
            3. Number each paper entry
            
            At the end, mention that the user can ask for more details about any specific paper.
            Use formal academic language throughout.
            """
            
            system_prompt = (
                "You are a professional academic research assistant. "
                "Your responses are clear, organized, and formatted for readability. "
                "You help researchers identify papers of interest from search results."
            )
            
            logger.info("[RESPONSE] Generating list all papers response")
            response = await self._model_provider.query(prompt, system_prompt)
            
            # Apply content filtering
            filtered_response = self._filter_inappropriate_content(response)
            
            # Log a preview of the response
            preview = filtered_response[:100] + "..." if len(filtered_response) > 100 else filtered_response
            logger.info(f"[RESPONSE] List papers response preview: '{preview}'")
            
            await final_response_stream.emit_chunk(filtered_response)
        except Exception as e:
            logger.error(f"[RESPONSE] Error generating list papers response: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback manual listing
            fallback_response = f"Here are the papers from your search on \"{previous_search_query}\":\n\n"
            
            for i, paper in enumerate(last_papers):
                fallback_response += f"{i+1}. \"{paper.get('title')}\" by {', '.join(paper.get('authors', [])[:2])}\n"
            
            fallback_response += "\nYou can ask for more details about any of these papers."
            
            logger.info("[RESPONSE] Using fallback list papers response")
            await final_response_stream.emit_chunk(fallback_response)
        
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] List papers response handling complete")
    
    async def _generate_search_summary(
            self,
            query: str,
            papers: List[Dict[str, Any]],
            author: str = None
    ) -> AsyncIterator[str]:
        """Generate a comprehensive summary of all search results."""
        context = f"search query: \"{query}\"" if query else "search"
        if author:
            context = f"search for papers by author \"{author}\""
            if query:
                context += f" related to \"{query}\""
        
        logger.info(f"[SUMMARY] Generating search summary for {context}")
        
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
        Based on the {context}, I found these papers on arXiv:
        
        {papers_text}
        
        Please provide:
        1. A brief overview of EACH paper found (1-2 sentences per paper)
        2. How they relate to the {context} (1-2 sentences)
        3. Any thematic connections between the papers (1-2 sentences)
        
        IMPORTANT: 
        - Cover ALL papers, not just the most relevant ones
        - Number each paper discussion (1-5)
        - Provide equal attention to each paper
        - Group papers by theme if possible
        - Use formal academic language throughout
        - DO NOT use informal language, slang, or profanity
        """
        
        # Custom system prompt for search summary with enhanced content filtering
        system_prompt = (
            "You are a professional academic research assistant specializing in scientific literature. "
            "You help researchers understand available papers on topics using formal, scholarly language. "
            "You provide comprehensive coverage of ALL papers found, not just the most relevant ones. "
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
            # Provide a simpler fallback response that still covers all papers
            logger.error(f"[SUMMARY] Error generating search summary: {str(e)}")
            logger.error(traceback.format_exc())
            
            logger.info("[SUMMARY] Using fallback summary response")
            
            author_context = f" by {author}" if author else ""
            query_context = f" on \"{query}\"" if query else ""
            
            simple_summary = f"Based on your search{author_context}{query_context}, I found {len(papers)} relevant papers:\n\n"
            yield simple_summary
            
            for i, paper in enumerate(papers):
                paper_summary = f"{i+1}. \"{paper['title']}\" by {', '.join(paper['authors'][:2])}\n"
                paper_summary += f"   Published: {paper['published']}\n"
                paper_summary += f"   Categories: {', '.join(paper['categories'])}\n"
                paper_summary += f"   Summary: {paper['summary'][:150]}...\n\n"
                yield paper_summary
            
            yield "You can ask for more details about any of these papers."
            logger.info("[SUMMARY] Fallback summary generation complete")

    async def _handle_author_profile_response(
            self,
            result: Dict[str, Any],
            response_handler: ResponseHandler,
            session: Session
    ):
        """Handle the response for an author profile query."""
        logger.info("[RESPONSE] Handling author profile response")
        
        author_name = result["author"]
        found = result["found"]
        
        # Notify user about fetching the author profile
        await response_handler.emit_text_block(
            "FETCH_AUTHOR", f"Fetching profile for author {author_name}..."
        )
        
        if found:
            profile = result["profile"]
            logger.info(f"[RESPONSE] Author profile found for: '{author_name}'")
            
            # Emit author profile as JSON
            await response_handler.emit_json(
                "AUTHOR_PROFILE", {"profile": profile}
            )
            
            try:
                # Generate and stream author profile summary
                logger.info("[RESPONSE] Generating author profile summary")
                summary_stream = response_handler.create_text_stream("AUTHOR_SUMMARY")
                async for chunk in self._generate_author_summary(profile):
                    filtered_chunk = self._filter_inappropriate_content(chunk)
                    await summary_stream.emit_chunk(filtered_chunk)
                await summary_stream.complete()
                logger.info("[RESPONSE] Author profile summary generation complete")
            except Exception as e:
                logger.error(f"[RESPONSE] Error generating author summary: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Create final response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            try:
                # Generate a detailed, conversational profile
                prompt = f"""
                Create a detailed profile summary for author {author_name} based on their publications.
                
                Author information:
                - Name: {author_name}
                - Total publications found: {profile['paper_count']}
                - Top research areas: {', '.join([item['category'] for item in profile['research_interests'][:3]])}
                - Publication timeline: {json.dumps(profile['publication_timeline'])}
                - Recent co-authors: {', '.join(profile['co_authors'][:5])}
                
                Recent papers (up to 5):
                {json.dumps([{
                    "title": paper.get("title"),
                    "published": paper.get("published"),
                    "categories": paper.get("categories", []),
                    "summary": paper.get("summary", "")[:150] + "..."
                } for paper in profile['papers'][:5]], indent=2)}
                
                Make your response:
                1. Start with a brief professional introduction of the author based on their research focus
                2. Describe their research interests and how they've evolved over time (if visible in the timeline)
                3. Mention their publication activity and recent work
                4. Highlight 2-3 of their most recent or significant papers with brief descriptions
                5. Mention key collaborators if notable patterns exist
                6. Conclude with a summary of their contribution to their field
                7. End by asking if the user would like to explore any specific paper
                
                Use formal academic language, be objective and thorough.
                """
                
                system_prompt = (
                    "You are a professional academic research assistant specializing in author profiles. "
                    "Your summaries provide comprehensive, objective overviews of researchers based on "
                    "their publication history. You identify patterns in their work and highlight their "
                    "key contributions to their field. Your tone is formal, scholarly, and precise."
                )
                
                logger.info(f"[RESPONSE] Calling model for author profile with prompt of length {len(prompt)}")
                response = await self._model_provider.query(prompt, system_prompt)
                
                # Apply content filtering
                filtered_response = self._filter_inappropriate_content(response)
                
                logger.info(f"[RESPONSE] Generated author profile of length {len(filtered_response)}")
                
                await final_response_stream.emit_chunk(filtered_response)
            except Exception as e:
                logger.error(f"[RESPONSE] Error generating author profile: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Fallback response
                fallback_response = f"## Author Profile: {author_name}\n\n"
                
                # Research interests
                fallback_response += "### Research Interests\n"
                for interest in profile['research_interests'][:5]:
                    fallback_response += f"- {interest['category']} ({interest['count']} papers)\n"
                
                # Publication history
                fallback_response += "\n### Publication History\n"
                for year_data in profile['publication_timeline']:
                    fallback_response += f"- {year_data['year']}: {year_data['papers']} publications\n"
                
                # Recent papers
                fallback_response += "\n### Recent Publications\n"
                for i, paper in enumerate(profile['papers'][:5]):
                    fallback_response += f"{i+1}. \"{paper.get('title')}\" ({paper.get('published')})\n"
                    fallback_response += f"   Categories: {', '.join(paper.get('categories', ['N/A']))}\n\n"
                
                # Co-authors
                fallback_response += "\n### Key Collaborators\n"
                for coauthor in profile['co_authors'][:5]:
                    fallback_response += f"- {coauthor}\n"
                
                fallback_response += "\nYou can ask for more details about any of these papers or request additional information about this author's work."
                
                logger.info("[RESPONSE] Using fallback author profile")
                await final_response_stream.emit_chunk(fallback_response)
                
            await final_response_stream.complete()
        else:
            # Author not found
            error = result.get("error", "Unknown error")
            logger.error(f"[RESPONSE] Author not found: {error}")
            
            await response_handler.emit_text_block(
                "ERROR", f"Could not find information for author: {error}"
            )
            
            # Create a more helpful error response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            await final_response_stream.emit_chunk(
                f"I couldn't find detailed information for author '{author_name}'. This might be due to:"
                f"\n\n- Variant spellings of the author's name"
                f"\n- The author using a different name format in publications"
                f"\n- Limited publication history on arXiv"
                f"\n\nTry searching for papers by this author with a more general query like 'search for papers by {author_name}'."
            )
            
            await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Author profile response handling complete")

    async def _generate_author_summary(
            self,
            profile: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Generate a comprehensive summary of an author's research profile."""
        
        author_name = profile["name"]
        paper_count = profile["paper_count"]
        research_interests = profile["research_interests"]
        papers = profile["papers"]
        timeline = profile["publication_timeline"]
        
        logger.info(f"[SUMMARY] Generating author summary for {author_name}")
        
        # Extract research interest categories
        top_categories = [item['category'] for item in research_interests[:5]]
        
        # Format recent papers for the summary
        recent_papers_text = "\n\n".join([
            f"Paper {i+1}:\n"
            f"Title: {paper['title']}\n"
            f"Published: {paper['published']}\n"
            f"Categories: {', '.join(paper['categories'])}\n"
            f"Summary: {paper['summary'][:200]}..."
            for i, paper in enumerate(papers[:5])
        ])
        
        # Create publication timeline summary
        timeline_str = ", ".join([f"{item['year']}: {item['papers']} papers" for item in timeline])
        
        prompt = f"""
        Generate a comprehensive academic profile for author {author_name}.
        
        Profile data:
        - Total publications found: {paper_count}
        - Research areas: {', '.join(top_categories)}
        - Publication timeline: {timeline_str}
        
        Recent papers:
        {recent_papers_text}
        
        Please provide:
        1. An overview of the author's research focus and expertise
        2. Analysis of their research trajectory and evolution over time
        3. Description of their publication impact and contributions to the field
        4. Brief summary of their most recent or significant work
        
        Use formal academic language appropriate for researchers.
        """
        
        system_prompt = (
            "You are a professional academic research assistant specializing in researcher profiles. "
            "You analyze publication patterns to create insightful, objective summaries of "
            "researchers' work and contributions. Your summaries highlight research focus, "
            "methodological approaches, and key contributions. You maintain scholarly "
            "objectivity while providing valuable context about the researcher's significance."
        )
        
        logger.info(f"[SUMMARY] Sending prompt of length {len(prompt)} to model")
        
        try:
            async for chunk in self._model_provider.query_stream(prompt, system_prompt):
                filtered_chunk = self._filter_inappropriate_content(chunk)
                yield filtered_chunk
            logger.info("[SUMMARY] Successfully completed author summary generation")
        except Exception as e:
            # Fallback for error cases
            logger.error(f"[SUMMARY] Error generating author summary: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Provide a simpler fallback response
            yield f"## Author Profile: {author_name}\n\n"
            
            # Research focus
            yield f"{author_name} has published {paper_count} papers on arXiv"
            if top_categories:
                yield f", primarily in the areas of {', '.join(top_categories[:3])}.\n\n"
            else:
                yield ".\n\n"
            
            # Recent papers
            yield "### Recent Publications:\n\n"
            
            for i, paper in enumerate(papers[:3]):
                title = paper.get('title', 'Untitled')
                published = paper.get('published', 'Unknown date')
                
                yield f"**{i+1}. {title}** ({published})\n\n"
                
                summary = paper.get('summary', '')
                if summary:
                    short_summary = summary[:200] + "..." if len(summary) > 200 else summary
                    yield f"{short_summary}\n\n"
            
            yield "You can request more details about any of these papers or explore other aspects of this author's work."


# Modified DefaultServer implementation that properly manages sessions
class ImprovedDefaultServer(DefaultServer):
    """A DefaultServer that properly maintains client sessions."""
    
    def __init__(self, agent: AbstractAgent):
        super().__init__(agent)
        self.client_sessions = {}
    
    async def process_request(self, request_data: dict):
        """Process a request with proper session management."""
        try:
            # Log the raw request data
            print("\n" + "=" * 80)
            print("[SERVER] RAW REQUEST DATA:")
            print(json.dumps(request_data, indent=2, default=str))
            print("=" * 80 + "\n")
            
            # Extract request information
            processor_id = request_data.get('processor_id', 'unknown')
            activity_id = request_data.get('activity_id', 'unknown')
            request_id = request_data.get('request_id', 'unknown')
            interactions = request_data.get('interactions', [])
            
            # Check if we have any interactions
            if not interactions:
                return {
                    "error": "No interactions provided",
                    "status": "error",
                    "message": "Request must include interactions"
                }
            
            # Use activity_id as client_id for session management
            client_id = activity_id
            
            # Get or create session for this client
            if client_id in self.client_sessions:
                session = self.client_sessions[client_id]
            else:
                session = Session()
                session.metadata = {
                    'client_id': client_id,
                    'processor_id': processor_id,
                    'activity_id': activity_id,
                    'request_id': request_id,
                    'interactions': []
                }
                self.client_sessions[client_id] = session
            
            # Update session with current interactions
            if not hasattr(session, 'metadata'):
                session.metadata = {}
            session.metadata['interactions'] = session.metadata.get('interactions', []) + interactions
            
            # Create query with context
            query = Query(
                prompt=request_data.get('prompt', ''),
                context={
                    'client_id': client_id,
                    'processor_id': processor_id,
                    'activity_id': activity_id,
                    'request_id': request_id,
                    'interactions': interactions
                }
            )
            
            # Create response handler
            response_handler = ResponseHandler()
            
            # Process the request using the agent
            await self.agent.assist(session, query, response_handler)
            
            # Store updated session
            self.client_sessions[client_id] = session
            
            # Get the response
            response = response_handler.get_response()
            
            return response
            
        except Exception as e:
            # Return an error response
            return {
                "error": str(e),
                "status": "error",
                "message": "Failed to process request"
            }


if __name__ == "__main__":
    # Create an instance of the ArxivResearchAgent
    agent = ArxivResearchAgent(name="arXiv Research Agent")
    # Create a server to handle requests to the agent
    server = ImprovedDefaultServer(agent)
    # Run the server
    server.run()