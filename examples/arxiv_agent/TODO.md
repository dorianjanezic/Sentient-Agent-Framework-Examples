# ArXiv Agent TODO List

## Search Papers Enhancement

### Current Implementation
1. User submits a prompt
2. Agent classifies it as search_papers
3. System directly proceeds to search_papers tool

### Proposed Enhancement
1. User submits a prompt
2. Agent classifies it as search_papers
3. System enters confirm_search_prompt state
4. Store enhanced LLM search entities in session metadata:
   - Category
   - Topic
   - Search terms
   - Other relevant entities
5. Present search entities to user
6. Allow user to:
   - Modify search entities
   - Confirm search entities
7. Proceed to search_papers tool with confirmed/modified entities

### Implementation Plan
1. Add new state `confirm_search_prompt` to state machine
2. Modify session metadata structure to store search entities
3. Update UI to display and allow modification of search entities
4. Implement confirmation flow
5. Update search_papers tool to use confirmed entities

### User Experience Impact
- More control over search parameters
- Better understanding of how search is being performed
- Ability to refine search before execution
- More accurate and relevant search results

## Future Features

### Daily Paper Analysis
- Implement daily feed of X papers
- Automated analysis of all papers
- Summary generation
- Trend detection
- Topic clustering

### Search Results Enhancement
- Add sorting information to search results:
  - Relevance
  - Submission date
  - Other sorting criteria 