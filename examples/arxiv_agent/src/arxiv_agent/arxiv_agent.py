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
from datetime import datetime, timedelta

from openai import RateLimitError, APIError, APITimeoutError
import arxiv

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
    GENERAL_EXPLANATION = "general_explanation"  # Open-ended discussion
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
        Search for papers on arXiv.
        
        Args:
            query: Search query string
            session: Current session
            agent: ArxivResearchAgent instance
            author: Optional author filter
            topic: Optional topic/category filter
            date_range: Optional date range filter
            sort_by: Optional sort order
            category: Optional category filter
            
        Returns:
            Dict containing search results and metadata
        """
        logger.info(f"[TOOL] search_papers called with query: '{query}'")
        
        # Get the explanation from the query classification if available
        query_classification = session.metadata.get('last_classification', {})
        classification_explanation = query_classification.get('explanation', '')
        
        if classification_explanation:
            logger.info(f"[TOOL] Classification explanation: '{classification_explanation}'")
        
        # First, transform the query to better search terms using the LLM
        base_search_terms = ""
        if query:
            # Pass the classification explanation to the transform method
            base_search_terms = await agent._transform_query_to_search_terms(
                query=query, 
                classification_explanation=classification_explanation
            )
            logger.info(f"[TOOL] Transformed search query '{query}' into: '{base_search_terms}'")
            
            # If the query has been transformed to include date constraints, prefer date sorting
            # This respects the LLM's output as the source of truth
            if not sort_by and "submittedDate" in base_search_terms:
                sort_by = "lastUpdatedDate"
                logger.info(f"[TOOL] Using lastUpdatedDate sorting due to date constraint in transformed query")
        
        # Build search parameters
        search_params = []
        
        # Add the transformed query if it exists
        if base_search_terms:
            search_params.append(f"({base_search_terms})")
        
        # Add author filter if provided
        if author:
            # Format author for arXiv search
            formatted_author = f'au:"{author}"'
            search_params.append(formatted_author)
            logger.info(f"[TOOL] Added author filter: {formatted_author}")
        
        # Add category/topic filter if provided
        if topic or category:
            # Map friendly names to arXiv category codes
            category_map = {
                'ai': 'cs.AI',
                'machine learning': 'cs.LG',
                'deep learning': 'cs.LG',
                'neural networks': 'cs.LG',
                'computer vision': 'cs.CV',
                'nlp': 'cs.CL',
                'natural language processing': 'cs.CL',
                'robotics': 'cs.RO',
                'reinforcement learning': 'cs.LG',
                'llm': 'cs.LG',
                'language models': 'cs.LG',
                'large language models': 'cs.LG',
                'embeddings': 'cs.LG',
                'transformer': 'cs.LG',
                'attention': 'cs.LG',
                'quantum': 'quant-ph',
                'quantum computing': 'quant-ph',
                'quantum information': 'quant-ph',
                'quantum mechanics': 'quant-ph'
            }
            
            # Prioritize category over topic if both are present
            category_code = category if category else category_map.get(topic.lower() if topic else '', None)
            
            if category_code:
                # Format category for arXiv search
                formatted_category = f'cat:{category_code}'
                search_params.append(formatted_category)
                logger.info(f"[TOOL] Added category filter: {formatted_category}")
            else:
                logger.warning(f"[TOOL] Could not map topic '{topic}' to arXiv category")
        
        # Add date range filter if provided
        if date_range:
            # Convert friendly date range to arXiv format
            date_map = {
                "last_day": "submittedDate:[NOW-1DAY TO NOW]",
                "last_week": "submittedDate:[NOW-7DAYS TO NOW]",
                "last_month": "submittedDate:[NOW-1MONTH TO NOW]",
                "last_year": "submittedDate:[NOW-1YEAR TO NOW]"
            }
            if date_range in date_map:
                search_params.append(date_map[date_range])
                logger.info(f"[TOOL] Added date range filter: {date_range}")
            elif "to" in date_range.lower():
                try:
                    start_date, end_date = date_range.lower().split("to")
                    start_date = start_date.strip()
                    end_date = end_date.strip()
                    # Validate and format dates (implementation omitted)
                    search_params.append(f"submittedDate:[{start_date} TO {end_date}]")
                    logger.info(f"[TOOL] Added custom date range filter: {date_range}")
                except Exception as e:
                    logger.error(f"[TOOL] Error parsing custom date range: {str(e)}")
        
        # Combine all search parameters
        final_query = " AND ".join(search_params) if search_params else "*"
        logger.info(f"[TOOL] Final search query: '{final_query}'")
        
        try:
            # Log the sorting parameter being used
            sort_value = sort_by if sort_by else "relevance"
            logger.info(f"[TOOL] Using sort_by: '{sort_value}'")
            
            # Execute the search
            search_results = await agent._arxiv_provider.search(
                query=final_query,
                max_results=5,
                sort_by=sort_value
            )
            
            # If we got very few results, try a less restrictive search
            if len(search_results) < 2:
                logger.info("[TOOL] Few results found, trying less restrictive search")
                
                # Make search less restrictive by using just the key terms
                less_restrictive_terms = final_query
                
                # Replace exact phrase matching with regular term matching
                less_restrictive_terms = less_restrictive_terms.replace('ti:"', 'ti:')
                less_restrictive_terms = less_restrictive_terms.replace('abs:"', 'abs:')
                less_restrictive_terms = less_restrictive_terms.replace('"', ' ')
                
                # Simplify the query if it has multiple AND terms
                if less_restrictive_terms.count("AND") > 1:
                    # Keep only essential filters (author, category, date range)
                    parts = []
                    core_terms = []
                    
                    for part in less_restrictive_terms.split(" AND "):
                        if any(prefix in part for prefix in ['au:', 'cat:', 'submittedDate:']):
                            parts.append(part)
                        elif '(' in part and ')' in part:
                            # This is likely our transformed query, extract key terms
                            core_part = part.replace('(', '').replace(')', '')
                            term_parts = core_part.split(" OR ")
                            # Take a couple of key terms
                            for term in term_parts[:2]:
                                if not any(prefix in term for prefix in ['ti:', 'abs:']):
                                    core_terms.append(term)
                                else:
                                    # Extract just the term without the prefix
                                    term_clean = term.split(':', 1)[1] if ':' in term else term
                                    core_terms.append(term_clean)
                    
                    # Add the core terms back with simpler structure
                    if core_terms:
                        core_query = " OR ".join(core_terms)
                        parts.append(core_query)
                    
                    less_restrictive_terms = " AND ".join(parts)
                
                logger.info(f"[TOOL] Trying less restrictive search: '{less_restrictive_terms}'")
                
                less_restrictive_results = await agent._arxiv_provider.search(
                    query=less_restrictive_terms,
                    max_results=5,
                    sort_by=sort_value
                )
                
                if len(less_restrictive_results) > len(search_results):
                    search_results = less_restrictive_results
                    logger.info(f"[TOOL] Found {len(search_results)} papers with less restrictive search")
            
            # Generate a summary of the search results with the improved prompt
            papers_text = "\n".join([
                f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nPublished: {paper['published']}\nSummary: {paper['summary'][:200]}..."
                for paper in search_results
            ])
            
            task_instructions = """
            Summarize arXiv search results, emphasizing relevance to the query.
            Highlight thematic connections and suggest optional refinements.
            """
            
            user_prompt = f"""
            Query: {query}
            Parameters:
            - Author: {author or 'N/A'}
            - Topic: {topic or 'N/A'}
            - Date Range: {date_range or 'N/A'}
            - Category: {category or 'N/A'}
            - Sort By: {sort_by or 'relevance'}
            
            Papers:
            {papers_text}
            
            Provide:
            1. Brief summary of each paper (1-2 sentences)
            2. Relevance to the query
            3. Thematic connections
            4. Optional refinements (e.g., "focus on NLP", "recent papers")
            Number each summary.
            """
            
            summary = await agent._model_provider.query(user_prompt, task_instructions)
            filtered_summary = agent._filter_inappropriate_content(summary)
            
            # Store the results in session for future reference
            await agent._manage_session_state(
                session,
                "search",
                {"last_papers": search_results}
            )
            
            # Store search parameters in session metadata
            search_params = {
                "query": query,
                "author": author,
                "topic": topic,
                "date_range": date_range,
                "sort_by": sort_by,
                "category": category
            }
            await agent._manage_session_state(session, "update", {"last_search_params": search_params})
            
            # Return results with metadata and summary
            return {
                "query": query,
                "results": search_results,
                "summary": filtered_summary,
                "filters": search_params
            }
            
        except Exception as e:
            logger.error(f"[TOOL] Error executing search: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "query": query,
                "results": [],
                "summary": f"Error searching for papers: {str(e)}",
                "filters": {
                    "author": author,
                    "topic": topic,
                    "date_range": date_range,
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
            
            # Extract papers from response - search_response is already a list of papers
            author_papers = search_response
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
            
            # Generate a summary of the author profile with the improved prompt
            papers_text = "\n".join([
                f"Title: {paper['title']}\nPublished: {paper['published']}\nSummary: {paper['summary'][:150]}..."
                for paper in author_papers[:5]
            ])
            
            task_instructions = """
            Generate author profiles, focusing on research focus and publication trends.
            Suggest further exploration of papers or collaborators.
            """
            
            user_prompt = f"""
            Author: {author_name}
            Paper Count: {profile['paper_count']}
            Top Research Areas: {', '.join([item['category'] for item in profile['research_interests'][:3]])}
            Publication Timeline: {', '.join([f'{item["year"]}: {item["papers"]} papers' for item in profile['publication_timeline']])}
            Recent Papers:
            {papers_text}
            
            Provide:
            1. Research focus introduction
            2. Publication trends analysis
            3. Summary of 2-3 recent papers
            4. Key collaborators
            5. Field contributions
            6. Suggested exploration (papers, authors)
            """
            
            summary = await agent._model_provider.query(user_prompt, task_instructions)
            filtered_summary = agent._filter_inappropriate_content(summary)
            
            # Prepare result data
            result_data = {
                "type": "author_profile",
                "author": author_name,
                "found": True,
                "profile": profile,
                "summary": filtered_summary
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
        
        # First try to find the paper in last_papers if we have them
        if hasattr(session, "metadata") and "last_papers" in session.metadata:
            last_papers = session.metadata["last_papers"]
            logger.info(f"[TOOL] Checking {len(last_papers)} papers from last search")
            
            # Try to find a matching paper by ID or title
            for paper in last_papers:
                # Extract clean paper ID for comparison
                paper_id_clean = agent._extract_paper_id(paper)
                if paper_id_clean and paper_id_clean == paper_id:
                    logger.info(f"[TOOL] Found exact matching paper by ID in last_papers: '{paper.get('title')}'")
                    return await agent._generate_paper_response(paper, paper_id, session)
                
                # If paper_id looks like a title, try fuzzy matching
                if paper_id and not any(c in paper_id for c in ["/", "."]) and len(paper_id.split()) > 1:
                    paper_title = paper.get('title', '').lower()
                    search_title = paper_id.lower()
                    
                    # Check for exact match first
                    if search_title in paper_title:
                        logger.info(f"[TOOL] Found exact matching paper by title in last_papers: '{paper.get('title')}'")
                        return await agent._generate_paper_response(paper, paper.get('id', ''), session)
                    
                    # Try fuzzy matching if no exact match
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, search_title, paper_title).ratio()
                    if similarity > 0.8:  # 80% similarity threshold
                        logger.info(f"[TOOL] Found fuzzy matching paper by title in last_papers: '{paper.get('title')}' (similarity: {similarity:.2f})")
                        return await agent._generate_paper_response(paper, paper.get('id', ''), session)
        
        # If we didn't find it in last_papers and paper_id looks like a title, search for it
        if paper_id and not any(c in paper_id for c in ["/", "."]) and len(paper_id.split()) > 1:
            logger.info(f"[TOOL] Searching for paper by title: '{paper_id}'")
            try:
                # First try exact title search
                search_results = await agent._arxiv_provider.search(
                    f'ti:"{paper_id}"',
                    max_results=5,  # Increased from 1 to allow for fuzzy matching
                    sort_by="relevance"
                )
                
                if search_results and len(search_results) > 0:
                    # If we have multiple results, try to find the best match
                    best_match = None
                    best_similarity = 0.8  # Minimum similarity threshold
                    
                    for paper in search_results:
                        paper_title = paper.get('title', '').lower()
                        search_title = paper_id.lower()
                        
                        # Check for exact match first
                        if search_title in paper_title:
                            best_match = paper
                            best_similarity = 1.0
                            break
                        
                        # Try fuzzy matching
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, search_title, paper_title).ratio()
                        if similarity > best_similarity:
                            best_match = paper
                            best_similarity = similarity
                    
                    if best_match:
                        logger.info(f"[TOOL] Found matching paper by title search: '{best_match.get('title')}' (similarity: {best_similarity:.2f})")
                        return await agent._generate_paper_response(best_match, best_match.get('id', ''), session)
                    else:
                        # If no good match found, try a broader search
                        search_results = await agent._arxiv_provider.search(
                            paper_id,  # Use the title as a general search term
                            max_results=5,
                            sort_by="relevance"
                        )
                        
                        if search_results and len(search_results) > 0:
                            # Try fuzzy matching on the broader results
                            best_match = None
                            best_similarity = 0.8
                            
                            for paper in search_results:
                                paper_title = paper.get('title', '').lower()
                                search_title = paper_id.lower()
                                similarity = SequenceMatcher(None, search_title, paper_title).ratio()
                                if similarity > best_similarity:
                                    best_match = paper
                                    best_similarity = similarity
                            
                            if best_match:
                                logger.info(f"[TOOL] Found matching paper in broader search: '{best_match.get('title')}' (similarity: {best_similarity:.2f})")
                                return await agent._generate_paper_response(best_match, best_match.get('id', ''), session)
                
                # If no match found, check if this might be a book or non-arXiv publication
                if "introduction" in paper_id.lower() or "book" in paper_id.lower():
                    return {
                        "type": "paper_detail",
                        "paper_id": paper_id,
                        "found": False,
                        "error": "This appears to be a book or published work, not an arXiv paper. arXiv only contains academic papers and preprints. You may want to search for this work in a library catalog or academic database."
                    }
                else:
                    return {
                        "type": "paper_detail",
                        "paper_id": paper_id,
                        "found": False,
                        "error": "Could not find this paper on arXiv. This could be because:\n1. It's not an arXiv paper (e.g., it's a published book or article)\n2. The title might be slightly different\n3. It might be in a different repository\n\nTry searching for academic papers on this topic instead."
                    }
            except Exception as e:
                logger.error(f"[TOOL] Error searching for paper by title: {str(e)}")
        
        # If we didn't find it in last_papers or by title search, try to get it by ID
        try:
            # Only try to get by ID if it looks like an arXiv ID
            if any(c in paper_id for c in ["/", "."]):
                paper = await agent._arxiv_provider.get_paper_by_id(paper_id)
                if "error" in paper:
                    return {
                        "type": "paper_detail",
                        "paper_id": paper_id,
                        "found": False,
                        "error": paper["error"]
                    }
                    
                logger.info(f"[TOOL] Successfully retrieved paper by ID: '{paper.get('title')}'")
                return await agent._generate_paper_response(paper, paper_id, session)
            else:
                # Check if this might be a book or non-arXiv publication
                if "introduction" in paper_id.lower() or "book" in paper_id.lower():
                    return {
                        "type": "paper_detail",
                        "paper_id": paper_id,
                        "found": False,
                        "error": "This appears to be a book or published work, not an arXiv paper. arXiv only contains academic papers and preprints. You may want to search for this work in a library catalog or academic database."
                    }
                else:
                    return {
                        "type": "paper_detail",
                        "paper_id": paper_id,
                        "found": False,
                        "error": "Could not find this paper on arXiv. This could be because:\n1. It's not an arXiv paper (e.g., it's a published book or article)\n2. The title might be slightly different\n3. It might be in a different repository\n\nTry searching for academic papers on this topic instead."
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
        """Return information about the agent itself or session history."""
        logger.info(f"[TOOL] get_agent_info called with query: '{query}'")
        
        # Check if query is about session history or agent capabilities
        is_history_query = any(term in query.lower() for term in [
            "recent", "history", "last", "previous", "we discussed", "talked about", 
            "researched", "searched", "looked up"
        ])
        
        if is_history_query and hasattr(session, "metadata"):
            # Construct session history summary
            history_data = []
            
            if "last_action" in session.metadata:
                history_data.append(f"Last action: {session.metadata['last_action']}")
            
            # Include search history
            if "last_search_query" in session.metadata:
                history_data.append(f"Last search query: {session.metadata['last_search_query']}")
            
            # Include paper details
            if "current_paper" in session.metadata:
                paper = session.metadata["current_paper"]
                history_data.append(f"Last viewed paper: {paper.get('title', 'Unknown')}")
            
            # Include explanation history
            if "last_explanation_query" in session.metadata:
                history_data.append(f"Last discussed concept: {session.metadata['last_explanation_query']}")
            
            # Include author profile history
            if "current_author" in session.metadata:
                author = session.metadata["current_author"]
                history_data.append(f"Last viewed author: {author.get('name', 'Unknown')}")
            
            # Include recent papers
            if "last_papers" in session.metadata:
                papers = session.metadata["last_papers"]
                if papers:
                    paper_titles = [p.get('title', 'Unknown') for p in papers[:3]]
                    history_data.append(f"Recent papers: {', '.join(paper_titles)}")
            
            # Create history prompt
            task_instructions = """
            Summarize the user's session history conversationally.
            Be helpful and informative.
            Suggest next steps based on their history.
            """
            
            # If we have history data, create a detailed prompt
            if history_data:
                user_prompt = f"""
                Query about session history: "{query}"
                
                Session History:
                {chr(10).join(f"- {item}" for item in history_data)}
                
                Provide:
                1. A summary of what the user has explored so far
                2. Suggestions for next steps (e.g., exploring a paper in more detail, related topics)
                3. A conversational response to their history query
                """
            else:
                # No history data available
                user_prompt = f"""
                Query about session history: "{query}"
                
                No significant session history is available yet. The user hasn't performed any searches or viewed any papers.
                
                Provide:
                1. A brief explanation that there's no significant history yet
                2. Suggestions for how they could start exploring research on arXiv
                3. A conversational response to their query
                """
            
            summary = await agent._model_provider.query(user_prompt, task_instructions)
            filtered_summary = agent._filter_inappropriate_content(summary)
            
            return {
                "type": "agent_info",
                "query": query,
                "summary": filtered_summary,
                "is_history_query": True
            }
        else:
            # Standard agent capabilities info
            task_instructions = """
            Explain the agent's capabilities conversationally, encouraging exploration.
            List capabilities clearly with examples.
            """
            
            user_prompt = f"""
            Query: {query}
            Agent Name: {agent.name}
            Description: An arXiv Research Agent for academic discussions and research assistance.
            Capabilities:
            - Discuss academic concepts and trends
            - Search for papers by topic
            - Summarize specific papers
            - Profile authors' research
            - Answer questions about capabilities
            - Track your research session history
            
            Provide:
            1. Query acknowledgment
            2. Capabilities list (numbered)
            3. Example queries (e.g., "What is AI?", "Find papers on quantum computing")
            """
            
            summary = await agent._model_provider.query(user_prompt, task_instructions)
            filtered_summary = agent._filter_inappropriate_content(summary)
            
            return {
                "type": "agent_info",
                "query": query,
                "summary": filtered_summary,
                "is_history_query": False
            }
    
    @staticmethod
    async def explain_concept(
        query: str,
        session: Session,
        agent: 'ArxivResearchAgent'
    ) -> Dict[str, Any]:
        """
        Handle open-ended discussions and explanations of academic concepts.
        
        Args:
            query: The question or concept to explain
            session: Session information
            agent: The ArxivResearchAgent instance
            
        Returns:
            Dictionary with explanation and optional related papers
        """
        logger.info(f"[TOOL] explain_concept called with query: '{query}'")
        
        # Get the explanation from the query classification if available
        query_classification = session.metadata.get('last_classification', {})
        classification_explanation = query_classification.get('explanation', '')
        
        if classification_explanation:
            logger.info(f"[TOOL] Classification explanation: '{classification_explanation}'")
        
        # First, transform the user query into better search terms using the LLM
        # Pass the classification explanation to the transform method
        search_terms = await agent._transform_query_to_search_terms(
            query=query,
            classification_explanation=classification_explanation
        )
        logger.info(f"[TOOL] Transformed query '{query}' into search terms: '{search_terms}'")
        
        task_instructions = """
        Engage in academic discussions about concepts, providing clear explanations.
        Only reference papers provided in the prompt, which are sourced from arXiv.
        If no arXiv papers are available, state this explicitly and avoid citing other sources.
        When referencing papers, use the exact titles provided in the prompt.
        Do not list the papers at the end of your response - they will be added automatically.
        Encourage further conversation or refinement.
        """
        
        user_prompt = f"""
        Query: {query}
        
        Provide:
        1. Concise explanation of the concept or topic (100-200 words)
        2. Key aspects, applications, or trends
        3. Connection to recent arXiv research (if papers are provided)
        4. Invitation to discuss further or refine the topic
        
        Important: 
        - Only reference papers that are explicitly provided in the prompt.
        - If no papers are provided, do not cite any specific papers in your response.
        - Do not list the papers at the end of your response - they will be added automatically.
        - Focus on integrating paper references naturally into your explanation.
        - Avoid redundant paper listings - papers will be shown separately.
        """
        
        # Always search for related papers to ensure we have arXiv sources
        search_results = []
        try:
            # Determine sort method based on the transformed query
            sort_method = "relevance"
            
            # If the transformed query includes a date constraint, use date-based sorting
            if "submittedDate" in search_terms:
                sort_method = "lastUpdatedDate"
                logger.info(f"[TOOL] Using date-based sorting due to date constraint in transformed query")
            
            logger.info(f"[TOOL] Final search terms: '{search_terms}'")
            logger.info(f"[TOOL] Using sort method: '{sort_method}'")
            
            # Try the transformed search terms first
            search_results = await agent._arxiv_provider.search(
                query=search_terms,  # Use transformed search terms instead of raw query
                max_results=5,  # Increased from 3 to 5
                sort_by=sort_method
            )
            
            # If we got very few results, try a less restrictive search
            if len(search_results) < 2:
                logger.info("[TOOL] Few results found, trying less restrictive search")
                
                # Make search less restrictive - remove exact phrase requirements and some constraints
                less_restrictive_terms = search_terms
                
                # Replace exact phrase matching with regular term matching
                less_restrictive_terms = less_restrictive_terms.replace('ti:"', 'ti:')
                less_restrictive_terms = less_restrictive_terms.replace('abs:"', 'abs:')
                less_restrictive_terms = less_restrictive_terms.replace('"', ' ')
                
                # Remove some AND constraints if there are multiple
                if less_restrictive_terms.count("AND") > 1:
                    # Keep only the most important parts - category and date if present
                    parts = []
                    for part in less_restrictive_terms.split(" AND "):
                        if "cat:" in part or "submittedDate" in part or "(" not in part:
                            parts.append(part)
                    
                    # Ensure we have at least one non-date, non-category constraint
                    if not any(p for p in parts if "cat:" not in p and "submittedDate" not in p):
                        # Find the first meaningful part
                        for part in less_restrictive_terms.split(" AND "):
                            if "cat:" not in part and "submittedDate" not in part:
                                parts.append(part)
                                break
                    
                    less_restrictive_terms = " AND ".join(parts)
                
                logger.info(f"[TOOL] Trying less restrictive search: '{less_restrictive_terms}'")
                
                less_restrictive_results = await agent._arxiv_provider.search(
                    query=less_restrictive_terms,
                    max_results=5,
                    sort_by=sort_method
                )
                
                if len(less_restrictive_results) > len(search_results):
                    search_results = less_restrictive_results
                    logger.info(f"[TOOL] Found {len(search_results)} papers with less restrictive search")
            
            # Add paper information to the prompt if we found relevant papers
            if search_results:
                papers_text = "\n".join([
                    f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nPublished: {paper['published']}\nSummary: {paper['summary'][:150]}..."
                    for paper in search_results
                ])
                user_prompt += f"\nRelated Papers from arXiv:\n{papers_text}"
                logger.info(f"[TOOL] Added {len(search_results)} related papers to the prompt")
            else:
                # If still no papers, use a fallback approach with basic terms
                logger.info("[TOOL] No papers found, trying fallback search with basic terms")
                
                # Extract core concepts from the query
                core_concepts = []
                for word in query.lower().split():
                    if len(word) > 3 and word not in ["what", "where", "when", "which", "find", "tell", "show", "give", "list", "most", "recent", "latest", "current"]:
                        core_concepts.append(word)
                
                # Use only the most important keywords
                if len(core_concepts) >= 2:
                    fallback_query = " ".join(core_concepts[:3])  # Use top 3 keywords with OR logic
                    logger.info(f"[TOOL] Trying fallback search with: '{fallback_query}'")
                    
                    fallback_results = await agent._arxiv_provider.search(
                        query=fallback_query,
                        max_results=5,
                        sort_by=sort_method
                    )
                    
                    if fallback_results:
                        search_results = fallback_results
                        papers_text = "\n".join([
                            f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nPublished: {paper['published']}\nSummary: {paper['summary'][:150]}..."
                            for paper in search_results
                        ])
                        user_prompt += f"\nRelated Papers from arXiv:\n{papers_text}"
                        logger.info(f"[TOOL] Added {len(search_results)} papers from fallback search")
                    else:
                        user_prompt += "\nNo relevant arXiv papers found for this query."
                        logger.info("[TOOL] No relevant papers found with fallback search")
                else:
                    user_prompt += "\nNo relevant arXiv papers found for this query."
                    logger.info("[TOOL] No relevant papers found for the query")
        except Exception as e:
            logger.error(f"[TOOL] Error searching for related papers: {str(e)}")
            user_prompt += "\nUnable to retrieve arXiv papers due to an error."
        
        # Generate explanation
        explanation = await agent._model_provider.query(user_prompt, task_instructions)
        filtered_explanation = agent._filter_inappropriate_content(explanation)
        
        # Validate references in explanation
        if search_results:
            # Check if any paper titles from search_results are referenced
            paper_titles = [paper['title'] for paper in search_results]
            referenced_papers = [title for title in paper_titles if title in filtered_explanation]
            
            if not referenced_papers:
                filtered_explanation += "\n\nNote: This explanation is based on general knowledge. While I found some relevant arXiv papers, I've chosen to provide a general explanation without specific paper references. You can ask me to discuss any of the found papers in detail."
            else:
                filtered_explanation += "\n\nNote: All referenced papers are from arXiv."
        else:
            filtered_explanation += "\n\nNote: This explanation is based on general knowledge. No relevant arXiv papers were found for this query. For arXiv-specific research, please refine your query."
        
        # Store related papers in session if we found any
        if search_results:
            await agent._manage_session_state(
                session,
                "explanation",
                {
                    "query": query,
                    "related_papers": search_results
                }
            )
        
        return {
            "query": query,
            "explanation": filtered_explanation,
            "related_papers": search_results
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
        model_name = os.getenv("MODEL_NAME", "accounts/fireworks/models/llama-v3p3-70b-instruct")
        
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
            if data:
                if 'last_papers' in data:
                    session.metadata['last_papers'] = data['last_papers']
                    logger.info(f"[SESSION] Stored {len(data['last_papers'])} papers in session")
                if 'query' in data:
                    session.metadata['last_search_query'] = data['query']
                    session.metadata['last_search_time'] = datetime.now().isoformat()
                session.metadata['last_action'] = 'search'
        elif action == "paper_detail":
            if data and 'paper_id' in data:
                session.metadata['last_viewed_paper'] = data['paper_id']
                session.metadata['last_viewed_time'] = datetime.now().isoformat()
                if 'paper' in data:
                    session.metadata['current_paper'] = data['paper']
                session.metadata['last_action'] = 'paper_detail'
        elif action == "author_profile":
            if data and 'author' in data:
                session.metadata['last_viewed_author'] = data['author']
                session.metadata['last_viewed_time'] = datetime.now().isoformat()
                if 'profile' in data:
                    session.metadata['current_author'] = data['profile']
                session.metadata['last_action'] = 'author_profile'
        elif action == "explanation":
            if data and 'query' in data:
                session.metadata['last_explanation_query'] = data['query']
                session.metadata['last_explanation_time'] = datetime.now().isoformat()
                if 'related_papers' in data and data['related_papers']:
                    session.metadata['last_papers'] = data['related_papers']
                session.metadata['last_action'] = 'explanation'
        elif action == "update":
            if data:
                for key, value in data.items():
                    session.metadata[key] = value
                    logger.info(f"[SESSION] Updated {key} in session metadata")
        
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
            
            # Ensure session metadata is properly initialized
            if not hasattr(session, 'metadata') or not session.metadata:
                session.metadata = {}
                logger.info("[ASSIST] Initializing session metadata")
            
            # Get activity_id from session object or metadata
            activity_id = None
            if hasattr(session, '_session_object'):
                activity_id = getattr(session._session_object, 'activity_id', None)
                if activity_id:
                    session.metadata['activity_id'] = str(activity_id)
                    logger.info(f"[ASSIST] Set activity_id from session object: {activity_id}")
            
            if not activity_id:
                activity_id = session.metadata.get('activity_id')
                if not activity_id:
                    logger.warning("[ASSIST] No activity_id found in session metadata")
                    await self._handle_error_response(
                        "Session initialization error: No activity_id found",
                        response_handler
                    )
                    return
            
            # Restore session metadata from global store if available
            if activity_id in self._session_store:
                session.metadata.update(self._session_store[activity_id])
                logger.info(f"[ASSIST] Restored metadata from global store for activity {activity_id}")
            
            # Initialize required metadata fields if not present
            if 'interactions' not in session.metadata:
                session.metadata['interactions'] = []
            if 'conversation_history' not in session.metadata:
                session.metadata['conversation_history'] = []
            
            # Update session in global store
            self._session_store[activity_id] = session.metadata
            logger.info(f"[ASSIST] Updated global store for activity {activity_id}")
            
            # Now proceed with query classification and handling
            query_type, query_data = await self._classify_query(query.prompt, session)
            logger.info(f"[ASSIST] Query classified as {query_type.value} with data: {query_data}")
            
            # Store the classification results in session for context in other methods
            session.metadata['last_classification'] = query_data
            
            # Get the tool name to use
            tool_name = query_data.get("tool", "none")
            logger.info(f"[ASSIST] Selected tool: {tool_name}")
            
            # Handle the query based on tool name
            if tool_name == "search_papers":
                result = await Tools.search_papers(
                    query=query_data.get('query', query.prompt),  # Use extracted query with fallback
                    session=session,
                    agent=self,
                    author=query_data.get('author'),
                    topic=query_data.get('topic'),
                    date_range=query_data.get('date_range'),
                    sort_by=query_data.get('sort_by'),
                    category=query_data.get('category')
                )
                await self._handle_search_response(result, response_handler, session)
                
            elif tool_name == "get_paper_details":
                paper_id = query_data.get('paper_id')
                paper_title = query_data.get('paper_title')
                
                # Also check for 'title' since the LLM sometimes uses this parameter name
                if not paper_title and 'title' in query_data:
                    paper_title = query_data.get('title')
                    logger.info(f"[ASSIST] Found title parameter instead of paper_title: '{paper_title}'")
                
                # If we have a paper title but no paper ID, use the title
                if not paper_id and paper_title:
                    paper_id = paper_title
                    logger.info(f"[ASSIST] Using paper title for search: '{paper_id}'")
                
                # If we still don't have a paper_id, try to extract it from the query directly
                if not paper_id and '"' in query.prompt:
                    # Try to extract title from quotes in the query
                    match = re.search(r'"([^"]+)"', query.prompt)
                    if match:
                        paper_id = match.group(1)
                        logger.info(f"[ASSIST] Extracted paper title from quotes: '{paper_id}'")
                
                if not paper_id:
                    await self._handle_error_response(
                        "Please provide a paper ID or reference a paper from previous results.",
                        response_handler
                    )
                    return
                
                result = await Tools.get_paper_details(paper_id, session, self)
                await self._handle_paper_detail_response(result, response_handler, session)
                
            elif tool_name == "get_author_profile":
                author_name = query_data.get('author')
                if not author_name:
                    await self._handle_error_response(
                        "Please provide an author name to search for.",
                        response_handler
                    )
                    return
                
                result = await Tools.get_author_profile(author_name, session, self)
                await self._handle_author_profile_response(result, response_handler, session)
                
            elif tool_name == "explain_concept":
                result = await Tools.explain_concept(
                    query_data.get('query', query.prompt),
                    session,
                    self
                )
                await self._handle_explanation_response(result, response_handler, session)
                
            elif tool_name == "get_agent_info":
                result = await Tools.get_agent_info(query.prompt, session, self)
                await self._handle_agent_info_response(result, response_handler)
                
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
                
            elif query_type == QueryType.GREETING:
                await self._handle_greeting_response(query.prompt, response_handler)
                
            else:
                # Create a more helpful error response for unknown queries
                final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
                
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
                
                # Use LLM to generate a more helpful response
                try:
                    task_instructions = """
                    Provide helpful guidance when queries are unclear.
                    Suggest potential intents and ways to rephrase for better results.
                    """
                    
                    user_prompt = f"""
                    The user's query was unclear: "{query.prompt}"
                    
                    Conversation context:
                    {context}
                    
                    Generate a helpful response that:
                    1. Acknowledges the query was unclear
                    2. Analyzes the possible intent
                    3. Provides specific suggestions
                    4. Gives clear examples of how to rephrase
                    5. Keeps the response concise (2-3 sentences)
                    6. Uses formal academic language
                    7. Maintains a helpful and encouraging tone
                    """
                    
                    logger.info("[RESPONSE] Generating unknown query response with LLM")
                    response = await self._model_provider.query(user_prompt, task_instructions)
                    
                    await final_response_stream.emit_chunk(response)
                    
                except Exception as e:
                    logger.error(f"[RESPONSE] Error generating LLM unknown query response: {str(e)}")
                    # Even in case of error, try one more time with a simpler prompt
                    try:
                        fallback_prompt = f"""
                        The user asked: "{query.prompt}"
                        
                        Generate a brief, helpful response explaining that you need more specific details to help them.
                        Keep it under 2 sentences and maintain a professional tone.
                        """
                        
                        fallback_response = await self._model_provider.query(fallback_prompt, "")
                        await final_response_stream.emit_chunk(fallback_response)
                    except Exception as e2:
                        logger.error(f"[RESPONSE] Error generating fallback response: {str(e2)}")
                        # If all else fails, use a very basic response
                        await final_response_stream.emit_chunk("I need more specific details to help you. Could you please rephrase your question?")
                
                await final_response_stream.complete()
            
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
        # Get activity_id from session
        activity_id = None
        if hasattr(session, '_session_object'):
            activity_id = getattr(session._session_object, 'activity_id', None)
            if activity_id:
                activity_id = str(activity_id)
        
        # Initialize or restore session metadata
        if not hasattr(session, 'metadata'):
            session.metadata = {}
            logger.info("[CLASSIFY] Initialized empty session metadata in _classify_query")
            
            # Restore metadata from global store if available
            if activity_id and activity_id in self._session_store:
                session.metadata.update(self._session_store[activity_id])
                logger.info(f"[CLASSIFY] Restored metadata from global store for activity {activity_id}")
        
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
                    if "last_search_query" in session.metadata:
                        context += f"\nThe search query was: {session.metadata['last_search_query']}"
                        
                elif session.metadata["last_action"] == "paper_detail" and "current_paper" in session.metadata:
                    paper = session.metadata["current_paper"]
                    context += f"\nThe current paper being discussed is: {paper.get('title', 'Unknown')}"
                    
                elif session.metadata["last_action"] == "author_profile" and "current_author" in session.metadata:
                    author = session.metadata["current_author"]
                    context += f"\nThe current author being discussed is: {author.get('name', 'Unknown')}"
                    
                elif session.metadata["last_action"] == "explanation" and "last_explanation_query" in session.metadata:
                    query = session.metadata["last_explanation_query"]
                    context += f"\nThe last concept discussed was: {query}"
        
        # Task instructions for query classification
        task_instructions = """
        Classify user queries to select appropriate tools or discussion modes.
        Prioritize GENERAL_EXPLANATION for vague, discussion-oriented, or definition queries.
        Detect multiple intents and focus on the primary one.
        Return ONLY valid JSON, no other text.
        Your explanation should be DETAILED and capture important signals like recency or specificity.
        """
        
        # User prompt for query classification with JSON output
        user_prompt = f"""
        Analyze the query: "{query_text}"
        
        Context: {context}
        
        Classify it into one of these types: SEARCH, PAPER_DETAIL, AUTHOR_PROFILE, FOLLOWUP, GENERAL_EXPLANATION, AGENT_INFO, GREETING, UNKNOWN.
        - SEARCH: Explicit paper search (e.g., "Find papers on machine learning")
        - PAPER_DETAIL: Specific paper details (e.g., "Summarize paper 2101.12345")
        - AUTHOR_PROFILE: Author information (e.g., "Tell me about John Doe")
        - FOLLOWUP: Refers to prior results (e.g., "More on paper #1")
        - GENERAL_EXPLANATION: Concept discussion or definition (e.g., "What is AI?")
        - AGENT_INFO: Agent capabilities or information about past activities/queries (e.g., "What can you do?", "What did we discuss last?", "What topics have I researched recently?")
        - GREETING: Simple greetings (e.g., "Hello")
        - UNKNOWN: Unclear queries
        
        Importantly:
        - Queries about past discussions, recent topics, or session history should be classified as AGENT_INFO
        - Queries containing "recent", "latest", "current", or "trending" but asking about academic topics (not user history) should be GENERAL_EXPLANATION
        
        Return ONLY a JSON object with these fields:
        - type: Query type (string)
        - tool: Tool to call (search_papers, get_author_profile, get_paper_details, explain_concept, get_agent_info, none)
        - parameters: Tool parameters (e.g., query, author, paper_id)
        - confidence: Score (0.0 to 1.0)
        - explanation: Classification rationale - MUST include details like:
          * Any recency signals ("user is asking about recent research")
          * Any specificity signals ("user wants specific details")
          * Any category/topic signals ("query relates to machine learning")
          * Any time period signals ("user wants papers from last year")
          * This explanation will be used for query transformation
        
        Example response:
        {{
            "type": "GENERAL_EXPLANATION",
            "tool": "explain_concept",
            "parameters": {{"query": "What is AI?"}},
            "confidence": 0.9,
            "explanation": "Query asks for a general explanation about what AI is, seeking a definition or overview rather than specific papers. No recency indicators present."
        }}
        
        Example with recency:
        {{
            "type": "GENERAL_EXPLANATION",
            "tool": "explain_concept",
            "parameters": {{"query": "What are recent advances in quantum computing?"}},
            "confidence": 0.9,
            "explanation": "Query asks about recent advances in quantum computing, indicating the user wants current research or developments in this field. The recency signal 'recent advances' suggests the need for up-to-date information."
        }}
        
        IMPORTANT: Return ONLY the JSON object, no other text or explanation.
        """
        
        try:
            response = await self._model_provider.query(user_prompt, task_instructions)
            
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            
            # Try to find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    result = json.loads(json_str)
                    
                    # Get query type from result
                    query_type_str = result.get("type", "UNKNOWN").upper()
                    try:
                        query_type = QueryType[query_type_str]
                    except (KeyError, AttributeError):
                        logger.warning(f"[CLASSIFY] Invalid query type: {query_type_str}, defaulting to UNKNOWN")
                        query_type = QueryType.UNKNOWN
                    
                    # Get parameters from result
                    entities = result.get("parameters", {})
                    
                    # Add other result data to entities
                    entities["tool"] = result.get("tool", "none")
                    entities["confidence"] = result.get("confidence", 0.0)
                    entities["explanation"] = result.get("explanation", "Unknown reason")
                    
                    logger.info(f"[CLASSIFY] LLM classification result: {result}")
                    return query_type, entities
                except json.JSONDecodeError as je:
                    logger.error(f"[CLASSIFY] JSON decode error: {str(je)}")
                    logger.error(f"[CLASSIFY] Invalid JSON string: {json_str}")
            else:
                logger.error(f"[CLASSIFY] No JSON object found in response: {response}")
                
            # If we get here, something went wrong with JSON parsing
            logger.warning("[CLASSIFY] Falling back to pattern-based classification")
            return await self._classify_query_pattern_based(query_text, session)
                
        except Exception as e:
            logger.error(f"[CLASSIFY] LLM classification failed: {str(e)}")
            logger.error(traceback.format_exc())
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
        text = query_text.lower().strip()
        entities = {"query": query_text}
        
        # Check for greetings
        greeting_patterns = ["hello", "hi ", "hey", "greetings", "what's up"]
        for pattern in greeting_patterns:
            if text.startswith(pattern) or pattern.strip() == text:
                logger.info("[CLASSIFY] Matched greeting pattern")
                entities["tool"] = "none"
                return QueryType.GREETING, entities
        
        # Check for agent info queries
        agent_info_patterns = ["who are you", "what can you do", "what are you", "help", "your capabilities"]
        if any(pattern in text for pattern in agent_info_patterns):
            logger.info(f"[CLASSIFY] Matched agent info pattern: '{next((p for p in agent_info_patterns if p in text), None)}'")
            entities["tool"] = "get_agent_info"
            return QueryType.AGENT_INFO, entities
        
        # Check for paper detail patterns
        paper_detail_patterns = [
            "paper details", "details about paper", "tell me about paper", 
            "summarize paper", "paper info", "paper summary", "tell me more about paper",
            "explain paper", "analyze paper", "paper number", "paper #"
        ]
        
        arxiv_id_pattern = r'(\d{4}\.\d{5})(v\d+)?'
        arxiv_url_pattern = r'arxiv\.org\/abs\/(\d{4}\.\d{5})(v\d+)?'
        old_arxiv_pattern = r'(\w+)\/(\d{7})(v\d+)?'  # e.g., physics/0609047
        
        # Check for paper detail patterns or paper IDs
        if any(pattern in text for pattern in paper_detail_patterns):
            # Check for numerical paper reference
            paper_number_match = re.search(r'paper\s+(?:number\s+)?#?(\d+)', text)
            if paper_number_match and "last_papers" in session.metadata:
                paper_number = int(paper_number_match.group(1)) - 1  # Convert to 0-based index
                papers = session.metadata["last_papers"]
                if 0 <= paper_number < len(papers):
                    paper = papers[paper_number]
                    entities["paper_id"] = self._extract_paper_id(paper)
                    logger.info(f"[CLASSIFY] Found paper by number {paper_number + 1}: {entities['paper_id']}")
                    entities["tool"] = "get_paper_details"
                    return QueryType.PAPER_DETAIL, entities
            
            # Extract paper title if mentioned
            for pattern in paper_detail_patterns:
                if pattern in text:
                    parts = text.split(pattern, 1)
                    if len(parts) > 1:
                        paper_title = parts[1].strip()
                        entities["paper_title"] = paper_title
                        entities["tool"] = "get_paper_details"
                        logger.info(f"[CLASSIFY] Paper detail request for title: '{paper_title}'")
                        return QueryType.PAPER_DETAIL, entities
        
        # Check for arxiv IDs directly
        arxiv_id_match = re.search(arxiv_id_pattern, query_text)
        arxiv_url_match = re.search(arxiv_url_pattern, query_text)
        old_arxiv_match = re.search(old_arxiv_pattern, query_text)
        
        if arxiv_id_match or arxiv_url_match or old_arxiv_match:
            if arxiv_id_match:
                paper_id = arxiv_id_match.group(1)
            elif arxiv_url_match:
                paper_id = arxiv_url_match.group(1)
            else:  # old_arxiv_match
                category = old_arxiv_match.group(1)
                id_num = old_arxiv_match.group(2)
                paper_id = f"{category}/{id_num}"
                
            entities["paper_id"] = paper_id
            entities["tool"] = "get_paper_details"
            logger.info(f"[CLASSIFY] Matched paper ID pattern: '{paper_id}'")
            return QueryType.PAPER_DETAIL, entities
        
        # Check for author profile patterns
        author_patterns = [
            "author profile", "papers by author", "author search", "find author", 
            "search author", "who is author", "about author", "publications by"
        ]
        
        for pattern in author_patterns:
            if pattern in text:
                parts = text.split(pattern, 1)
                if len(parts) > 1:
                    author_name = parts[1].strip()
                    if author_name.startswith("by "):
                        author_name = author_name[3:].strip()
                    
                    entities["author"] = author_name
                    entities["tool"] = "get_author_profile"
                    logger.info(f"[CLASSIFY] Author profile request for: '{author_name}'")
                    return QueryType.AUTHOR_PROFILE, entities
        
        # Check for general explanation patterns
        explanation_patterns = [
            "what is", "explain", "how does", "tell me about", "define", 
            "describe", "discuss", "elaborate on", "can you explain"
        ]
        
        for pattern in explanation_patterns:
            if text.startswith(pattern) or f" {pattern} " in f" {text} ":
                # Extract the concept to explain
                concept = None
                if text.startswith(pattern):
                    concept = text[len(pattern):].strip()
                else:
                    parts = text.split(pattern, 1)
                    if len(parts) > 1:
                        concept = parts[1].strip()
                
                if concept:
                    entities["query"] = concept
                    entities["tool"] = "explain_concept"
                    logger.info(f"[CLASSIFY] General explanation request for: '{concept}'")
                    return QueryType.GENERAL_EXPLANATION, entities
        
        # Check for search patterns
        search_patterns = [
            "find papers", "search for papers", "look for papers", 
            "papers about", "papers on", "research on", "articles about"
        ]
        
        for pattern in search_patterns:
            if pattern in text:
                parts = text.split(pattern, 1)
                if len(parts) > 1:
                    search_term = parts[1].strip()
                    entities["query"] = search_term
                    entities["tool"] = "search_papers"
                    logger.info(f"[CLASSIFY] Search request for: '{search_term}'")
                    return QueryType.SEARCH, entities
        
        # Check for follow-up queries if we have session context
        if hasattr(session, "metadata") and "last_action" in session.metadata:
            followup_patterns = [
                "tell me more", "elaborate", "explain further", "details", "more information",
                "continue", "expand", "what about", "follow up", "additional info"
            ]
            
            if any(pattern in text for pattern in followup_patterns):
                logger.info(f"[CLASSIFY] Matched follow-up pattern with session context")
                return QueryType.FOLLOWUP, {}
        
        # If no specific pattern matched but query has academic or research terms,
        # classify as GENERAL_EXPLANATION
        academic_terms = [
            "concept", "theory", "definition", "field", "research", "study",
            "science", "technique", "method", "approach", "academic", "principle"
        ]
        
        if any(term in text for term in academic_terms):
            logger.info(f"[CLASSIFY] Detected academic term, classifying as general explanation")
            entities["query"] = query_text
            entities["tool"] = "explain_concept"
            return QueryType.GENERAL_EXPLANATION, entities
        
        # Default to treating as a search query
        entities["query"] = query_text
        entities["tool"] = "search_papers"
        logger.info(f"[CLASSIFY] Default classification as search: '{query_text}'")
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
        search_results = result["results"]
        filters = result.get("filters", {})
        summary = result.get("summary", "")
        
        # Notify user about the search
        search_message = f"Searching arXiv for papers related to '{query}'"
        if filters.get("author"):
            search_message += f" by author {filters['author']}"
        if filters.get("topic"):
            search_message += f" in category {filters['topic']}"
        if filters.get("date_range"):
            search_message += f" from {filters['date_range']}"
        search_message += "..."
        
        await response_handler.emit_text_block(
            "SEARCH", search_message
        )
        
        if len(search_results) > 0:
            # Emit search results as JSON
            await response_handler.emit_json(
                "PAPERS", {"results": search_results}
            )
            logger.info(f"[RESPONSE] Emitted {len(search_results)} papers as JSON")
            
            # Create final response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # If we have a summary from the result, use it
            if summary:
                await final_response_stream.emit_chunk(summary)
            else:
                # Generate our own response
                formatted_response = f"I found {len(search_results)} papers related to '{query}':\n\n"
                
                for i, paper in enumerate(search_results):
                    formatted_response += f"{i+1}. \"{paper['title']}\" by {', '.join(paper['authors'][:2])}"
                    if len(paper['authors']) > 2:
                        formatted_response += " et al."
                    formatted_response += f" ({paper['published'].split('T')[0]})\n"
                    
                    # Add a brief summary
                    summary = paper.get('summary', '')
                    if summary:
                        summary_snippet = summary[:150] + "..." if len(summary) > 150 else summary
                        formatted_response += f"   Summary: {summary_snippet}\n"
                    
                    formatted_response += "\n"
                
                formatted_response += "You can ask for more details about any specific paper by referring to its number or title."
                
                await final_response_stream.emit_chunk(formatted_response)
            
            await final_response_stream.complete()
        else:
            # No results found
            await response_handler.emit_text_block(
                "NO_RESULTS", f"No papers found matching '{query}'"
            )
            
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # Suggest alternatives
            suggestion_message = f"I couldn't find any papers matching '{query}'"
            if filters.get("author") or filters.get("topic") or filters.get("date_range"):
                suggestion_message += ". Consider trying again with fewer filters"
                if filters.get("author"):
                    suggestion_message += ", or check the spelling of the author name"
                if filters.get("topic"):
                    suggestion_message += ", or try a different category"
            
            suggestion_message += ". You could also try using more general keywords or a related research area."
            
            await final_response_stream.emit_chunk(suggestion_message)
            await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Search response complete")
    
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
            summary = result.get("summary", "")
            
            logger.info(f"[RESPONSE] Paper found: '{paper.get('title')}'")
            
            # Emit paper details as JSON
            await response_handler.emit_json(
                "PAPER_DETAILS", {"paper": paper}
            )
            
            # Create final response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # If we have a summary, use it; otherwise create our own response
            if summary:
                await final_response_stream.emit_chunk(summary)
            else:
                # Extract paper details
                title = paper['title']
                authors = ', '.join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors += " et al."
                published = paper['published'].split('T')[0]
                categories = ', '.join(paper['categories'])
                abstract = paper.get('summary', '')
                
                formatted_response = f"## {title}\n\n"
                formatted_response += f"**Authors:** {authors}\n"
                formatted_response += f"**Published:** {published}\n"
                formatted_response += f"**Categories:** {categories}\n\n"
                formatted_response += f"**Abstract:**\n{abstract}\n\n"
                
                # Add paper URLs if available
                if 'arxiv_url' in paper:
                    formatted_response += f"**arXiv URL:** {paper['arxiv_url']}\n"
                if 'pdf_url' in paper:
                    formatted_response += f"**PDF URL:** {paper['pdf_url']}\n\n"
                
                formatted_response += "You can ask me questions about this paper or explore other papers by these authors."
                
                await final_response_stream.emit_chunk(formatted_response)
            
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
            
            error_message = f"I couldn't find the paper with ID '{paper_id}'. "
            error_message += "This could be due to an incorrect ID format or the paper might not be available on arXiv. "
            error_message += "Please check the ID and try again, or try searching for the paper by its title."
            
            await final_response_stream.emit_chunk(error_message)
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
        
        # Emit paper details as JSON again for context
        await response_handler.emit_json(
            "PAPER_DETAILS", {"paper": paper}
        )
        
        # Create final response
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        try:
            # Track paper interactions in session metadata
            paper_id = self._extract_paper_id(paper)
            
            if not hasattr(session, "metadata"):
                session.metadata = {}
            
            # Initialize or get paper interaction history
            paper_interactions = session.metadata.get("paper_interactions", {})
            
            # Get the interaction count and history for this paper
            if paper_id not in paper_interactions:
                paper_interactions[paper_id] = {
                    "count": 0,
                    "queries": []
                }
            
            # Update interaction count and add this query
            paper_interactions[paper_id]["count"] += 1
            paper_interactions[paper_id]["queries"].append(query_text)
            interaction_count = paper_interactions[paper_id]["count"]
            previous_queries = paper_interactions[paper_id]["queries"][:-1]  # All except current
            
            # Store back in session
            session.metadata["paper_interactions"] = paper_interactions
            
            # Update session in global store
            await self._update_session(session)
            
            logger.info(f"[RESPONSE] Paper {paper_id} interaction count: {interaction_count}")
            
            # Let the LLM determine how to handle this follow-up based on the full context
            task_instructions = """
            Analyze paper interaction history to provide tailored follow-up responses.
            For repeat interactions, provide progressively deeper analysis without repetition.
            For first-time or specific questions, focus on directly answering the query.
            """
            
            # First determine the response strategy with the LLM
            strategy_prompt = f"""
            User query: "{query_text}"
            
            Paper information:
            Title: {paper.get('title')}
            Authors: {', '.join(paper.get('authors', []))}
            Published: {paper.get('published').split('T')[0]}
            Categories: {', '.join(paper.get('categories', []))}
            
            Interaction history with this paper:
            - Times this paper has been discussed: {interaction_count}
            - Previous queries about this paper: {previous_queries if previous_queries else "None"}
            
            ANALYZE this situation and determine the appropriate response strategy:
            1. If this is a REPEAT generic request for information about a paper already discussed, respond with STRATEGY: DEEPER_ANALYSIS
            2. If this is a SPECIFIC question needing a direct answer, respond with STRATEGY: DIRECT_ANSWER
            3. If this is the FIRST interaction with this paper, respond with STRATEGY: INITIAL_SUMMARY
            
            Return ONLY the strategy identifier, no other text.
            """
            
            strategy = await self._model_provider.query(strategy_prompt, task_instructions)
            strategy = strategy.strip().upper()
            
            logger.info(f"[RESPONSE] LLM-determined strategy: {strategy}")
            
            # Generate response based on the determined strategy
            if "DEEPER_ANALYSIS" in strategy:
                user_prompt = f"""
                The user is asking for more information about this paper:
                Title: {paper.get('title')}
                Authors: {', '.join(paper.get('authors', []))}
                Published: {paper.get('published').split('T')[0]}
                Categories: {', '.join(paper.get('categories', []))}
                Abstract: {paper.get('summary', '')}
                
                Their question is: "{query_text}"
                
                IMPORTANT: This is interaction #{interaction_count} with this paper.
                Previous queries: {previous_queries}
                
                Provide a DEEPER ANALYSIS that includes:
                1. Methodology and technical approach analysis
                2. Implications and applications of this research
                3. Relationships to broader research fields
                4. Critical analysis of strengths/limitations
                5. Specific follow-up questions the user could ask
                
                AVOID repeating information already likely covered in previous interactions.
                Focus on providing new insights and perspectives.
                Use scholarly language and academic reasoning.
                """
            elif "DIRECT_ANSWER" in strategy:
                user_prompt = f"""
                The user is asking a specific question about this paper:
                Title: {paper.get('title')}
                Authors: {', '.join(paper.get('authors', []))}
                Published: {paper.get('published').split('T')[0]}
                Categories: {', '.join(paper.get('categories', []))}
                Abstract: {paper.get('summary', '')}
                
                Their specific question is: "{query_text}"
                
                Please provide:
                1. A direct answer to their specific question
                2. Only include context from the paper if directly relevant to the question
                3. Be concise and scholarly in your response
                """
            else:  # Default to INITIAL_SUMMARY
                user_prompt = f"""
                The user is asking about this paper:
                Title: {paper.get('title')}
                Authors: {', '.join(paper.get('authors', []))}
                Published: {paper.get('published').split('T')[0]}
                Categories: {', '.join(paper.get('categories', []))}
                Abstract: {paper.get('summary', '')}
                
                Their question is: "{query_text}"
                
                Since this is the first detailed discussion of this paper, please provide:
                1. A concise overview of the paper's main contributions
                2. Brief context about the research area
                3. How this paper relates to the field
                4. Suggestions for further exploration
                """
            
            response = await self._model_provider.query(user_prompt, task_instructions)
            filtered_response = self._filter_inappropriate_content(response)
            
            await final_response_stream.emit_chunk(filtered_response)
        except Exception as e:
            logger.error(f"[RESPONSE] Error generating follow-up response: {str(e)}")
            
            # Fallback response
            title = paper.get('title', 'this paper')
            authors = ', '.join(paper.get('authors', [])[:2])
            if len(paper.get('authors', [])) > 2:
                authors += " et al."
            abstract = paper.get('summary', 'No abstract available.')
            
            fallback_message = f"Regarding your question about \"{title}\" by {authors}:\n\n"
            fallback_message += f"The paper abstract mentions: {abstract[:200]}...\n\n"
            fallback_message += "For more specific details, you may want to read the full paper or ask a more specific question."
            
            await final_response_stream.emit_chunk(fallback_message)
        
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Follow-up response handling complete")
    
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
            summary = result.get("summary", "")
            
            logger.info(f"[RESPONSE] Author profile found for: '{author_name}'")
            
            # Emit author profile as JSON
            await response_handler.emit_json(
                "AUTHOR_PROFILE", {"profile": profile}
            )
            
            # Create final response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            # If we have a summary, use it; otherwise create our own response
            if summary:
                await final_response_stream.emit_chunk(summary)
            else:
                # Extract profile details
                paper_count = profile.get('paper_count', 0)
                research_interests = [item.get('category') for item in profile.get('research_interests', [])[:3]]
                papers = profile.get('papers', [])
                co_authors = profile.get('co_authors', [])[:5]
                timeline = profile.get('publication_timeline', [])
                
                # Format the response
                formatted_response = f"## Author Profile: {author_name}\n\n"
                formatted_response += f"**Papers on arXiv:** {paper_count}\n"
                
                if research_interests:
                    formatted_response += f"**Main research areas:** {', '.join(research_interests)}\n\n"
                
                if timeline:
                    formatted_response += "**Publication Timeline:**\n"
                    timeline_str = ", ".join([f"{item.get('year')}: {item.get('papers')} papers" for item in timeline])
                    formatted_response += f"{timeline_str}\n\n"
                
                if papers:
                    formatted_response += "**Recent Papers:**\n"
                    for i, paper in enumerate(papers[:3]):
                        formatted_response += f"{i+1}. \"{paper.get('title')}\" ({paper.get('published').split('T')[0]})\n"
                
                if co_authors:
                    formatted_response += f"\n**Key Collaborators:** {', '.join(co_authors)}\n\n"
                
                formatted_response += "You can ask for more details on any of the papers or explore specific research areas."
                
                await final_response_stream.emit_chunk(formatted_response)
            
            await final_response_stream.complete()
        else:
            # Author not found
            error = result.get("error", "Unknown error")
            logger.error(f"[RESPONSE] Author not found: {error}")
            
            await response_handler.emit_text_block(
                "ERROR", f"Could not find author profile: {error}"
            )
            
            # Create a more helpful error response
            final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            
            error_message = f"I couldn't find an author profile for '{author_name}'. "
            error_message += "This could be due to different name formats used by the author in publications or limited presence on arXiv. "
            error_message += "Try variations of the name format or search for papers by this author with a more general query."
            
            await final_response_stream.emit_chunk(error_message)
            await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Author profile response handling complete")
    
    async def _handle_agent_info_response(
            self,
            result: Dict[str, Any],
            response_handler: ResponseHandler
    ):
        """Handle responses about the agent itself."""
        logger.info("[RESPONSE] Handling agent info response")
        
        # Create final response
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        # If we have a summary, use it; otherwise create our own response
        if "summary" in result:
            if result.get("is_history_query", False):
                # For history queries, add a header
                await final_response_stream.emit_chunk("## Your Research Session History\n\n")
            await final_response_stream.emit_chunk(result["summary"])
        else:
            # Create a default agent info response
            agent_info = """
            # arXiv Research Agent

            I'm an AI assistant specialized in helping you navigate academic research from arXiv. I can:

            1. Search for papers on specific topics
            2. Provide detailed summaries and information about specific papers
            3. Find and analyze an author's research profile and publications
            4. Explain academic concepts and provide context for research fields
            5. Answer follow-up questions about papers and research topics
            6. Track your research session and summarize what you've explored

            Just ask me to find papers on a topic you're interested in, get details about a specific paper, or learn about a researcher's work. I can also discuss general academic concepts to help you understand research better.
            """
            
            await final_response_stream.emit_chunk(agent_info)
        
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Agent info response handling complete")
    
    async def _handle_greeting_response(
            self,
            greeting: str,
            response_handler: ResponseHandler
    ):
        """Handle greeting responses."""
        logger.info("[RESPONSE] Handling greeting response")
        
        # Create final response
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        # Generate a friendly greeting
        try:
            task_instructions = """
            Respond to greetings with a friendly, professional welcome that introduces your capabilities.
            Keep responses concise and encourage research exploration.
            """
            
            user_prompt = f"""
            The user has greeted you with: "{greeting}"
            
            Generate a friendly, professional response that:
            1. Acknowledges their greeting
            2. Briefly introduces that you are an arXiv research assistant
            3. Mentions 2-3 key things you can help with
            4. Invites them to start researching
            5. Keeps the tone scholarly but conversational
            """
            
            response = await self._model_provider.query(user_prompt, task_instructions)
            filtered_response = self._filter_inappropriate_content(response)
            
            await final_response_stream.emit_chunk(filtered_response)
        except Exception as e:
            logger.error(f"[RESPONSE] Error generating greeting response: {str(e)}")
            
            # Fallback greeting
            greeting_resp = "Hello! I'm your arXiv research assistant. I can help you search for academic papers, get details about specific research, and explore authors' publication history. What topic would you like to explore today?"
            
            await final_response_stream.emit_chunk(greeting_resp)
        
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Greeting response handling complete")
        
    async def _handle_explanation_response(
            self,
            result: Dict[str, Any],
            response_handler: ResponseHandler,
            session: Session
    ):
        """Handle responses for general explanations and academic discussions."""
        logger.info("[RESPONSE] Handling explanation response")
        
        query = result["query"]
        explanation = result["explanation"]
        related_papers = result.get("related_papers", [])
        
        # Emit any related papers as JSON if available
        if related_papers:
            await response_handler.emit_json(
                "RELATED_PAPERS", {"results": related_papers}
            )
            logger.info(f"[RESPONSE] Emitted {len(related_papers)} related papers as JSON")
        
        # Create final response
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        # Use the explanation from the result
        await final_response_stream.emit_chunk(explanation)
        
        # If we have related papers, add a note about them
        if related_papers:
            paper_note = "\n\nRelated papers from arXiv:\n\n"
            
            for i, paper in enumerate(related_papers[:3]):
                paper_note += f"{i+1}. \"{paper.get('title')}\" by {', '.join(paper.get('authors', [])[:2])}"
                if len(paper.get('authors', [])) > 2:
                    paper_note += " et al."
                paper_note += f" ({paper.get('published', '').split('T')[0]})\n"
            
            paper_note += "\nYou can ask for more details about any of these papers if you're interested."
            
            await final_response_stream.emit_chunk(paper_note)
        
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Explanation response handling complete")

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
        
        # Create final response
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")
        
        # Create a more helpful error message
        try:
            task_instructions = """
            Create helpful, friendly error messages that guide users toward successful interactions.
            Provide suggestions for alternative approaches.
            """
            
            user_prompt = f"""
            There was an error with the user's request: "{error_message}"
            
            Generate a helpful response that:
            1. Acknowledges the issue in a professional way
            2. Explains what might have gone wrong
            3. Provides 1-2 specific suggestions for resolving it
            4. Maintains a scholarly but encouraging tone
            """
            
            response = await self._model_provider.query(user_prompt, task_instructions)
            filtered_response = self._filter_inappropriate_content(response)
            
            await final_response_stream.emit_chunk(filtered_response)
        except Exception as e:
            logger.error(f"[RESPONSE] Error generating error response: {str(e)}")
            
            # Fallback error message
            error_resp = f"I encountered a problem: {error_message}. Please try phrasing your request differently or try a different search term."
            
            await final_response_stream.emit_chunk(error_resp)
        
        await final_response_stream.complete()
        
        # Mark response as complete
        await response_handler.complete()
        logger.info("[RESPONSE] Error response handling complete")
    
    async def _generate_author_summary(
            self,
            profile: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Generate a comprehensive summary of an author's research profile."""
        author_name = profile.get("name", "")
        papers = profile.get("papers", [])
        research_interests = profile.get("research_interests", [])
        publication_timeline = profile.get("publication_timeline", [])
        co_authors = profile.get("co_authors", [])
        
        logger.info(f"[SUMMARY] Generating author summary for {author_name}")
        
        # Build a prompt for the LLM
        task_instructions = """
        Generate comprehensive academic author profiles based on publication data.
        Focus on research trends, contributions, and scholarly connections.
        """
        
        user_prompt = f"""
        Create a detailed academic profile for {author_name} based on these publications:
        
        Papers ({len(papers)} total):
        {json.dumps([{
            "title": p.get("title", ""),
            "published": p.get("published", ""),
            "categories": p.get("categories", []),
            "summary": p.get("summary", "")[:100] + "..." if p.get("summary") and len(p.get("summary", "")) > 100 else p.get("summary", "")
        } for p in papers[:5]], indent=2)}
        
        Research Areas:
        {json.dumps(research_interests, indent=2)}
        
        Publication Timeline:
        {json.dumps(publication_timeline, indent=2)}
        
        Collaborators:
        {json.dumps(co_authors[:5], indent=2)}
        
        Include:
        1. Research focus overview
        2. Publication trend analysis
        3. Key papers and contributions
        4. Research evolution over time (if evident)
        5. Collaborative network insights
        6. Field impact assessment
        
        Use formal academic language appropriate for research contexts.
        """
        
        try:
            # Call the model to generate author summary
            async for chunk in self._model_provider.query_stream(user_prompt, task_instructions):
                filtered_chunk = self._filter_inappropriate_content(chunk)
                yield filtered_chunk
            
            logger.info(f"[SUMMARY] Successfully generated author summary for {author_name}")
        except Exception as e:
            logger.error(f"[SUMMARY] Error generating author summary: {str(e)}")
            
            # Fallback response
            yield f"# {author_name}: Research Profile\n\n"
            
            # Research focus
            if research_interests:
                interests = [f"{item.get('category')} ({item.get('count')} papers)" for item in research_interests[:3]]
                yield f"## Research Focus\n\n{author_name} primarily publishes in {', '.join(interests)}.\n\n"
            
            # Recent papers
            if papers:
                yield "## Notable Publications\n\n"
                for i, paper in enumerate(papers[:3]):
                    yield f"**{i+1}. {paper.get('title')}** ({paper.get('published').split('T')[0]})\n\n"
                    
                    summary = paper.get('summary', '')
                    if summary:
                        brief_summary = summary[:150] + "..." if len(summary) > 150 else summary
                        yield f"{brief_summary}\n\n"
            
            # Timeline
            if publication_timeline:
                yield "## Publication History\n\n"
                history = ", ".join([f"{item.get('year')}: {item.get('papers')} papers" for item in publication_timeline])
                yield f"{history}\n\n"
            
            # Collaborators
            if co_authors:
                yield f"## Key Collaborators\n\n{', '.join(co_authors[:5])}\n\n"
            
            yield "You can ask for more details about any specific paper or research area."
            
            logger.info(f"[SUMMARY] Generated fallback author summary for {author_name}")

    async def _generate_paper_response(self, paper: Dict[str, Any], paper_id: str, session: Session) -> Dict[str, Any]:
        """Helper method to generate paper response with summary."""
        task_instructions = """
        Provide detailed paper information, focusing on contributions and context.
        Suggest related exploration options.
        """
        
        user_prompt = f"""
        Paper ID: {paper_id}
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Published: {paper['published']}
        Categories: {', '.join(paper['categories'])}
        Abstract: {paper['summary']}
        
        Provide:
        1. Brief introduction (1 sentence)
        2. Main contributions (1-2 sentences)
        3. Access links (arXiv URL, PDF)
        4. Suggested exploration (e.g., author's other papers, related topics)
        """
        
        summary = await self._model_provider.query(user_prompt, task_instructions)
        filtered_summary = self._filter_inappropriate_content(summary)
        
        # Prepare result data
        result_data = {
            "type": "paper_detail",
            "paper_id": paper_id,
            "paper": paper,
            "summary": filtered_summary,
            "found": True
        }
        
        # Update session state
        await self._manage_session_state(session, "paper_detail", result_data)
        
        return result_data

    async def _transform_query_to_search_terms(self, query: str, classification_explanation: str = None) -> str:
        """
        Transform a natural language query into effective search terms for arXiv
        using arXiv's advanced query syntax.
        
        Args:
            query: The original user query
            classification_explanation: Optional explanation from query classification
            
        Returns:
            Optimized search terms for arXiv API
        """
        logger.info(f"[TRANSFORM] Transforming query: '{query}'")
        
        # Skip transformation for direct paper IDs
        if re.search(r'\d{4}\.\d{5}', query):
            logger.info(f"[TRANSFORM] Query contains paper ID, using as is")
            return query
            
        # Get current date and one year ago for date filtering
        current_date = datetime.now().strftime("%Y%m%d")
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        
        task_instructions = """
        Convert natural language questions into effective arXiv API search terms.
        Your job is to create the most effective arXiv search string that will return relevant papers.
        This MUST include special handling for recency, ensuring all temporal queries return recent papers.
        """
        
        user_prompt = f"""
        Original query: "{query}"
        """
        
        # Add classification explanation context if available
        if classification_explanation:
            user_prompt += f"""
            Query classification: "{classification_explanation}"
            """
            logger.info(f"[TRANSFORM] Including classification explanation: '{classification_explanation}'")
        
        user_prompt += f"""
        Convert this query into effective search terms for the arXiv API using their specific query syntax.
        
        arXiv search field prefixes:
        - ti: for title search (e.g., ti:quantum)
        - au: for author search (e.g., au:smith)
        - abs: for abstract search (e.g., abs:machine learning)
        - co: for comment field search
        - jr: for journal reference search
        - cat: for category search (e.g., cat:cs.AI)
        - all: for searching all fields
        - submittedDate: for date range in format [YYYYMMDD TO YYYYMMDD]
        
        Boolean operators: AND, OR, ANDNOT (must be capitalized)
        Grouping: Use parentheses (e.g., (ti:quantum OR ti:algorithm) AND au:smith)
        
        IMPORTANT GUIDELINES:
        - For ANY query related to recent, latest, or current research:
          - ALWAYS include this date filter: submittedDate:[{one_year_ago} TO {current_date}]
        
        - Keep searches balanced between specific and broad:
          - Use OR between similar terms (e.g., quantum OR "quantum mechanics")
          - Use AND between different concepts (e.g., quantum AND computing)
          - Include both title (ti:) and abstract (abs:) searches for key terms
        
        - Format date ranges exactly as: submittedDate:[YYYYMMDD TO YYYYMMDD]
        
        Examples:
        Query: "What are recent advances in quantum computing?"
        Search terms: (ti:quantum OR abs:quantum) AND (ti:computing OR abs:computing) AND submittedDate:[{one_year_ago} TO {current_date}]
        
        Query: "Latest research on CRISPR gene editing"
        Search terms: (CRISPR OR "gene editing") AND submittedDate:[{one_year_ago} TO {current_date}]
        
        Query: "What is reinforcement learning?"
        Search terms: "reinforcement learning" AND (introduction OR overview OR survey)
        
        RETURN ONLY THE SEARCH TERMS. NO OTHER TEXT OR EXPLANATION.
        """
        
        try:
            response = await self._model_provider.query(user_prompt, task_instructions)
            
            # Clean the response
            response = response.strip()
            if response.lower().startswith("search terms:"):
                response = response[len("search terms:"):].strip()
                
            # Handle quotes properly
            response = response.replace('\\"', '"').replace("\\'", "'")
            
            # Ensure proper date format for submittedDate
            if "submittedDate:[" in response:
                # Remove any time components to keep format simple
                date_pattern = r'submittedDate:\[(\d{8})(\d{4})?(\s+TO\s+|\+TO\+)(\d{8})(\d{4})?\]'
                date_match = re.search(date_pattern, response)
                
                if date_match:
                    start_date = date_match.group(1)
                    end_date = date_match.group(4)
                    
                    # Reconstruct with standard format without time components
                    new_date_range = f"submittedDate:[{start_date} TO {end_date}]"
                    response = re.sub(date_pattern, new_date_range, response)
                
                # Replace any remaining space-based TO with standard format
                response = response.replace("+TO+", " TO ")
            
            # Ensure response isn't too long
            if len(response) > 500:
                logger.warning(f"[TRANSFORM] Transformed query too long: '{response}', trimming")
                response = response[:500]
            
            # If response is empty, use original query
            if not response:
                logger.warning(f"[TRANSFORM] Empty transformed query, using original")
                return query
                
            logger.info(f"[TRANSFORM] Successfully transformed query to: '{response}'")
            return response
        except Exception as e:
            logger.error(f"[TRANSFORM] Error transforming query: {str(e)}")
            return query  # Fall back to original query on error


# Modified DefaultServer implementation that properly manages sessions
class ImprovedDefaultServer(DefaultServer):
    """A DefaultServer that properly maintains client sessions and conversation history."""
    
    def __init__(self, agent: AbstractAgent):
        super().__init__(agent)
        self.client_sessions = {}
        
        # Add health check endpoint
        @self._app.get("/health")
        async def health_check():
            return {"status": "healthy"}
    
    async def process_request(self, request_data: dict):
        """Process a request with proper session management and conversation history."""
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
            
            # Get or create session for this activity_id
            if activity_id in self.client_sessions:
                session = self.client_sessions[activity_id]
                logger.info(f"[SESSION] Using existing session for activity {activity_id}")
            else:
                session = Session()
                logger.info(f"[SESSION] Creating new session for activity {activity_id}")
                self.client_sessions[activity_id] = session
            
            # Initialize session metadata from global store if available
            if activity_id in self.agent._session_store:
                session.metadata = self.agent._session_store[activity_id].copy()
                logger.info(f"[SESSION] Restored metadata from global store for activity {activity_id}")
            else:
                # Initialize new metadata
                session.metadata = {
                    'client_id': activity_id,
                    'processor_id': processor_id,
                    'activity_id': activity_id,
                    'request_id': request_id,
                    'interactions': [],
                    'conversation_history': []
                }
                logger.info(f"[SESSION] Initialized new metadata for activity {activity_id}")
            
            # Update conversation history with new interactions
            session.metadata['conversation_history'].extend(interactions)
            session.metadata['interactions'] = interactions  # Keep current interactions separate
            
            # Create query with full context including conversation history
            query = Query(
                prompt=request_data.get('prompt', ''),
                context={
                    'client_id': activity_id,
                    'processor_id': processor_id,
                    'activity_id': activity_id,
                    'request_id': request_id,
                    'interactions': interactions,
                    'conversation_history': session.metadata['conversation_history']
                }
            )
            
            # Create response handler
            response_handler = ResponseHandler()
            
            # Process the request
            await self.agent.assist(session, query, response_handler)
            
            # Update global session store with latest metadata
            self.agent._session_store[activity_id] = session.metadata
            logger.info(f"[SESSION] Updated global store for activity {activity_id}")
            
            # Get response
            response = response_handler.get_response()
            
            # Log the response
            print("\n" + "=" * 80)
            print("[SERVER] RESPONSE:")
            print(json.dumps(response, indent=2, default=str))
            print("=" * 80 + "\n")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "error": str(e),
                "status": "error",
                "message": "An error occurred while processing the request"
            }


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Create an instance of the ArxivResearchAgent
    agent = ArxivResearchAgent(name="arXiv Research Agent")
    # Create a server to handle requests to the agent
    server = ImprovedDefaultServer(agent)
    # Get the port from environment variable
    port = int(os.getenv("PORT", "8080"))
    # Run the server with Uvicorn on the specified port
    uvicorn.run(server._app, host="0.0.0.0", port=port)