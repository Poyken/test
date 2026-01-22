"""
AI Flow - PM Agent (Project Manager)
Extracts requirements and user stories from meeting notes
"""

from typing import List
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from orchestrator.state import WorkflowState, UserStory, TaskPriority


class UserStoriesOutput(BaseModel):
    """Output schema for user stories extraction"""
    stories: List[dict]


class PMAgent(BaseAgent):
    """
    Project Manager Agent
    
    Responsibilities:
    - Analyze meeting notes/documents
    - Extract business requirements
    - Create user stories with acceptance criteria
    - Prioritize and estimate complexity
    """
    
    @property
    def name(self) -> str:
        return "PM Agent"
    
    @property
    def system_prompt(self) -> str:
        return """You are a Senior Project Manager with 15 years of experience in software development.
Your role is to analyze meeting notes and project documents to extract clear, actionable requirements.

STRICT RULES:
1. Extract ONLY what is explicitly mentioned or strongly implied in the documents
2. Do NOT add features that are not mentioned
3. Each user story must follow the standard format
4. Acceptance criteria must be testable and specific
5. Priority is based on business value and dependencies
6. Complexity is based on technical effort (S=1-2 hours, M=2-8 hours, L=1-3 days, XL=3+ days)

OUTPUT FORMAT:
You must respond with a JSON object containing a "stories" array.
Each story must have these exact fields:
- id: string (format: "US-001", "US-002", etc.)
- title: string
- description: string
- role: string (the user role)
- feature: string (what they want)
- benefit: string (why they want it)
- acceptance_criteria: array of strings
- priority: string ("P0", "P1", "P2", or "P3")
- complexity: string ("S", "M", "L", or "XL")

Example output:
```json
{
  "stories": [
    {
      "id": "US-001",
      "title": "User Registration",
      "description": "Allow new users to create accounts",
      "role": "new customer",
      "feature": "register an account with email and password",
      "benefit": "I can access the platform and make purchases",
      "acceptance_criteria": [
        "User can enter email and password",
        "Email validation is performed",
        "Password must be at least 8 characters",
        "Confirmation email is sent"
      ],
      "priority": "P0",
      "complexity": "M"
    }
  ]
}
```"""
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Extract user stories from meeting notes"""
        self.log("Analyzing meeting notes and extracting user stories...")
        
        # Truncate meeting notes if too long to avoid token limits
        meeting_notes = state['meeting_notes']
        if len(meeting_notes) > 15000:
            meeting_notes = meeting_notes[:15000] + "\n\n[... truncated for brevity ...]"
        
        user_prompt = f"""Analyze the following project documentation and extract user stories.

PROJECT DOCUMENTATION:
{meeting_notes}

Extract the TOP 5 most important user stories for an MVP.
Focus on core e-commerce features: authentication, products, cart, orders, payments.

IMPORTANT: Return ONLY a valid JSON object. No markdown, no explanation.
The JSON must have this exact structure:
{{"stories": [
  {{"id": "US-001", "title": "...", "description": "...", "role": "...", "feature": "...", "benefit": "...", "acceptance_criteria": ["..."], "priority": "P0", "complexity": "M"}}
]}}"""

        try:
            response = await self.invoke_llm(user_prompt)
            
            # Parse the response with robust JSON extraction
            import json
            import re
            
            response_text = response
            
            # Try to extract JSON from various formats
            # Method 1: Look for ```json blocks
            if "```json" in response_text:
                match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            elif "```" in response_text:
                match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            
            # Method 2: Find JSON object directly
            if not response_text.strip().startswith('{'):
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            # Clean up common issues
            response_text = response_text.strip()
            
            # Try to parse
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                self.log(f"Attempting to repair JSON: {e}", "warning")
                
                # Remove trailing commas before ] or }
                response_text = re.sub(r',\s*([}\]])', r'\1', response_text)
                # Fix unescaped newlines in strings
                response_text = re.sub(r'(?<!\\)\n(?=[^"]*"[^"]*$)', r'\\n', response_text)
                
                try:
                    data = json.loads(response_text)
                except:
                    # Last resort: extract individual stories manually
                    self.log("Using fallback story extraction", "warning")
                    data = {"stories": []}
                    story_matches = re.findall(r'\{[^{}]*"id"\s*:\s*"US-\d+"[^{}]*\}', response_text)
                    for match in story_matches[:5]:
                        try:
                            story = json.loads(match)
                            data["stories"].append(story)
                        except:
                            continue
            
            stories_data = data.get("stories", [])
            
            # Convert to UserStory objects
            user_stories = []
            for s in stories_data:
                try:
                    story = UserStory(
                        id=s.get("id", f"US-{len(user_stories)+1:03d}"),
                        title=s.get("title", "Untitled"),
                        description=s.get("description", ""),
                        role=s.get("role", "user"),
                        feature=s.get("feature", s.get("title", "")),
                        benefit=s.get("benefit", ""),
                        acceptance_criteria=s.get("acceptance_criteria", []),
                        priority=TaskPriority(s.get("priority", "P1")),
                        complexity=s.get("complexity", "M"),
                    )
                    user_stories.append(story)
                except Exception as e:
                    self.log(f"Warning: Could not parse story: {e}", "warning")
                    continue
            
            self.log(f"Extracted {len(user_stories)} user stories", "success")
            
            # Update state
            state["user_stories"] = user_stories
            
        except Exception as e:
            self.log(f"Error extracting user stories: {e}", "error")
            state["errors"].append(f"PM Agent error: {str(e)}")
        
        return state

