"""
AI Flow - Architect Agent
Designs technical architecture from requirements
"""

from typing import List
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from orchestrator.state import WorkflowState, TechnicalSpec


class ArchitectAgent(BaseAgent):
    """
    Architect Agent
    
    Responsibilities:
    - Design technical architecture for each feature
    - Define database schema changes
    - Specify API endpoints
    - Plan frontend components
    - Identify dependencies and integration points
    """
    
    @property
    def name(self) -> str:
        return "Architect Agent"
    
    @property
    def system_prompt(self) -> str:
        return """You are a Senior Software Architect with deep expertise in:
- NestJS (Node.js backend framework)
- Next.js (React frontend framework)
- PostgreSQL with Prisma ORM
- Multi-tenant SaaS architecture
- RESTful API design

Your role is to translate user stories into technical specifications.

TECH STACK CONSTRAINTS:
- Backend: NestJS 11, TypeScript, Prisma 6.x
- Frontend: Next.js 16, React 19, TailwindCSS 4, Zustand
- Database: PostgreSQL 15+ with pgvector
- Cache: Redis 7+
- Queue: BullMQ

ARCHITECTURAL PATTERNS TO FOLLOW:
1. Multi-tenant: All tables have tenantId column, use AsyncLocalStorage for tenant context
2. Soft delete: All entities have deletedAt column
3. Audit trail: createdAt, updatedAt, createdBy, updatedBy
4. DTOs: Use class-validator for input validation
5. Services: Business logic in services, not controllers
6. Repository pattern: Database access through Prisma services

OUTPUT FORMAT:
Respond with a JSON object containing a "specs" array.
Each spec must have:
- feature_id: string (references the user story ID)
- database_changes: array of strings describing schema changes
- api_endpoints: array of strings in format "METHOD /path - description"
- frontend_components: array of strings describing React components needed
- dependencies: array of strings listing other features this depends on

Example:
```json
{
  "specs": [
    {
      "feature_id": "US-001",
      "database_changes": [
        "CREATE TABLE users (id, email, password_hash, tenantId, ...)",
        "CREATE INDEX idx_users_email ON users(email)"
      ],
      "api_endpoints": [
        "POST /api/v1/auth/register - Create new user account",
        "POST /api/v1/auth/login - Authenticate user"
      ],
      "frontend_components": [
        "RegisterForm - User registration form with validation",
        "LoginForm - Login form with email/password"
      ],
      "dependencies": []
    }
  ]
}
```"""
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Create technical specifications from user stories"""
        self.log("Designing technical architecture for user stories...")
        
        if not state.get("user_stories"):
            self.log("No user stories found, skipping architecture design", "warning")
            return state
        
        # Format user stories for the prompt (limit to 3 to avoid token limits)
        stories = state["user_stories"][:3]
        stories_text = "\n\n".join([
            f"""USER STORY: {s.id}
Title: {s.title}
As a {s.role}, I want {s.feature}"""
            for s in stories
        ])
        
        user_prompt = f"""Design technical specs for these user stories.

{stories_text}

Return ONLY a valid JSON object with this structure:
{{"specs": [
  {{"feature_id": "US-001", "database_changes": ["..."], "api_endpoints": ["POST /api/v1/..."], "frontend_components": ["..."], "dependencies": []}}
]}}

Keep each array short (max 3 items). No markdown, no explanation."""

        try:
            response = await self.invoke_llm(user_prompt)
            
            # Parse with robust JSON extraction
            import json
            import re
            
            response_text = response
            
            # Extract JSON
            if "```json" in response_text:
                match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            elif "```" in response_text:
                match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            
            if not response_text.strip().startswith('{'):
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            response_text = response_text.strip()
            
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.log(f"Attempting to repair JSON: {e}", "warning")
                response_text = re.sub(r',\s*([}\]])', r'\1', response_text)
                try:
                    data = json.loads(response_text)
                except:
                    self.log("Using fallback spec extraction", "warning")
                    data = {"specs": []}
            
            specs_data = data.get("specs", [])
            
            # Convert to TechnicalSpec objects
            technical_specs = []
            for s in specs_data:
                try:
                    spec = TechnicalSpec(
                        feature_id=s.get("feature_id", f"US-{len(technical_specs)+1:03d}"),
                        database_changes=s.get("database_changes", [])[:5],
                        api_endpoints=s.get("api_endpoints", [])[:5],
                        frontend_components=s.get("frontend_components", [])[:5],
                        dependencies=s.get("dependencies", []),
                    )
                    technical_specs.append(spec)
                except Exception as e:
                    self.log(f"Warning: Could not parse spec: {e}", "warning")
                    continue
            
            self.log(f"Created {len(technical_specs)} technical specifications", "success")
            
            # Update state
            state["technical_specs"] = technical_specs
            
        except Exception as e:
            self.log(f"Error creating technical specs: {e}", "error")
            state["errors"].append(f"Architect Agent error: {str(e)}")
        
        return state

