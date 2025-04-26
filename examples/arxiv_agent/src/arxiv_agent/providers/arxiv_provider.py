import arxiv
from typing import List, Dict, Any

class ArxivProvider:
    def __init__(self):
        """Initialize the arXiv provider."""
        # Configure the default arxiv client
        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=3,
            num_retries=3
        )
    
    async def search(
            self,
            query: str,
            max_results: int = 5,
            sort_by: str = "submitteddate"
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: How to sort results - "relevance", "lastUpdatedDate", or "submittedDate"
            
        Returns:
            Dictionary containing:
            - results: List of paper metadata dictionaries
            - total_count: Total number of matching papers (if available)
        """
        # Map sort_by string to arxiv library's SortCriterion
        sort_criterion = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
            "submitteddate": arxiv.SortCriterion.SubmittedDate
        }.get(sort_by.lower(), arxiv.SortCriterion.SubmittedDate)
        
        # Create the search query
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )
        
        # Execute the search and get results
        results = []
        for paper in self.client.results(search):
            # Convert each paper to a dictionary
            paper_dict = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "updated": paper.updated.strftime("%Y-%m-%d") if paper.updated else None,
                "doi": paper.doi,
                "journal_ref": paper.journal_ref,
                "pdf_url": paper.pdf_url,
                "primary_category": paper.primary_category,
                "categories": paper.categories,
                "comment": paper.comment,
                "id": paper.entry_id,
                "arxiv_url": paper.entry_id.replace("http://arxiv.org/", "https://arxiv.org/")
            }
            results.append(paper_dict)
        
        # Try to get total count of matching papers
        total_results_count = None
        try:
            # The search object might have metadata about total results
            total_results_count = search.total_results
        except AttributeError:
            # If not available directly, we can only report what we have
            total_results_count = len(results)
        
        return {
            "results": results,
            "total_count": total_results_count
        }
    
    async def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific paper by its arXiv ID.
        
        Args:
            paper_id: arXiv ID (e.g., "2101.12345")
            
        Returns:
            Paper metadata dictionary
        """
        # Clean the ID if it's a full URL
        if paper_id.startswith(("http://arxiv.org/", "https://arxiv.org/")):
            paper_id = paper_id.split("/")[-1]
        
        # Remove version information if present
        if "v" in paper_id:
            paper_id = paper_id.split("v")[0]
            
        # Create a search for the specific paper
        search = arxiv.Search(
            id_list=[paper_id],
            max_results=1
        )
        
        # Get the paper data
        results = list(self.client.results(search))
        if not results:
            return {"error": f"Paper with ID {paper_id} not found"}
        
        paper = results[0]
        
        # Convert paper to dictionary
        paper_dict = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "published": paper.published.strftime("%Y-%m-%d"),
            "updated": paper.updated.strftime("%Y-%m-%d") if paper.updated else None,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "pdf_url": paper.pdf_url,
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "comment": paper.comment,
            "id": paper.entry_id,
            "arxiv_url": paper.entry_id.replace("http://arxiv.org/", "https://arxiv.org/")
        }
        
        return paper_dict