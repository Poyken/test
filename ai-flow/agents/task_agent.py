"""
AI Flow - Task Agent
Breaks down features into atomic development tasks
"""

from typing import List
import json

from agents.base_agent import BaseAgent
from orchestrator.state import WorkflowState, Task, TaskStatus, TaskPriority, TaskType


class TaskAgent(BaseAgent):
    """
    Task Decomposition Agent
    
    Responsibilities:
    - Break down technical specs into atomic tasks
    - Define task dependencies
    - Assign tasks to appropriate agent types (backend, frontend, etc.)
    - Estimate effort for each task
    - Create ordered execution plan
    """
    
    @property
    def name(self) -> str:
        return "Task Agent"
    
    @property
    def system_prompt(self) -> str:
        return """You are a Technical Lead responsible for breaking down features into atomic development tasks.

TASK PRINCIPLES:
1. Each task should be completable in 15-60 minutes
2. Tasks must have clear, testable outcomes
3. Dependencies must be explicit - a task can only start after its dependencies are complete
4. Tasks should be typed correctly (backend, frontend, database, api, test, documentation)

TASK ORDERING RULES:
1. Database tasks come first (schema, migrations)
2. API/Backend tasks come after database
3. Frontend tasks can start after relevant API endpoints exist
4. Test tasks come after implementation tasks
5. Documentation tasks come last

OUTPUT FORMAT:
Respond with a JSON object containing a "tasks" array.
Each task must have:
- id: string (format: "TASK-001", "TASK-002", etc.)
- title: string (concise)
- description: string (detailed)
- type: string ("backend", "frontend", "database", "api", "test", "documentation")
- story_id: string (references the user story)
- dependencies: array of task IDs this depends on
- files_to_create: array of file paths to create
- files_to_modify: array of existing file paths to modify
- estimated_minutes: number (15-60 range preferred)
- priority: string ("P0", "P1", "P2", "P3")

Example:
```json
{
  "tasks": [
    {
      "id": "TASK-001",
      "title": "Add User model to Prisma schema",
      "description": "Add the User model with all required fields including tenantId, email, passwordHash, etc.",
      "type": "database",
      "story_id": "US-001",
      "dependencies": [],
      "files_to_create": [],
      "files_to_modify": ["prisma/schema.prisma"],
      "estimated_minutes": 15,
      "priority": "P0"
    },
    {
      "id": "TASK-002",
      "title": "Generate Prisma migration",
      "description": "Run prisma migrate dev to create the database migration for User model",
      "type": "database",
      "story_id": "US-001",
      "dependencies": ["TASK-001"],
      "files_to_create": ["prisma/migrations/xxx_add_user/migration.sql"],
      "files_to_modify": [],
      "estimated_minutes": 5,
      "priority": "P0"
    }
  ]
}
```"""
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Break down technical specs into atomic tasks"""
        self.log("Breaking down features into atomic tasks...")
        
        if not state.get("technical_specs"):
            self.log("No technical specs found, skipping task breakdown", "warning")
            return state
        
        import re
        
        # Simplified specs for prompt
        specs = state["technical_specs"][:2]  # Limit to 2 specs
        specs_text = "\n".join([
            f"Feature {spec.feature_id}: {', '.join(spec.api_endpoints[:2])}"
            for spec in specs
        ])
        
        user_prompt = f"""Create 5 development tasks for these features:
{specs_text}

Return ONLY valid JSON:
{{"tasks": [
  {{"id": "TASK-001", "title": "...", "description": "...", "type": "database", "story_id": "US-001", "dependencies": [], "files_to_create": [], "files_to_modify": [], "estimated_minutes": 30, "priority": "P0"}}
]}}

Types: database, backend, api, frontend, test
No markdown, no explanation."""

        try:
            response = await self.invoke_llm(user_prompt)
            
            # Robust JSON extraction
            response_text = response
            
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
                    self.log("Using fallback - empty tasks", "warning")
                    data = {"tasks": []}
            
            tasks_data = data.get("tasks", [])
            
            # Convert to Task objects
            tasks = []
            for i, t in enumerate(tasks_data[:10]):  # Limit to 10 tasks
                try:
                    task_type = t.get("type", "backend")
                    if task_type not in ["backend", "frontend", "database", "api", "test", "documentation"]:
                        task_type = "backend"
                    
                    task = Task(
                        id=t.get("id", f"TASK-{i+1:03d}"),
                        title=t.get("title", "Untitled task"),
                        description=t.get("description", ""),
                        type=TaskType(task_type),
                        story_id=t.get("story_id", "US-001"),
                        dependencies=t.get("dependencies", []),
                        files_to_create=t.get("files_to_create", []),
                        files_to_modify=t.get("files_to_modify", []),
                        estimated_minutes=t.get("estimated_minutes", 30),
                        priority=TaskPriority(t.get("priority", "P1")),
                        status=TaskStatus.PENDING,
                    )
                    tasks.append(task)
                except Exception as e:
                    self.log(f"Warning: Could not parse task: {e}", "warning")
                    continue
            
            # Create topologically sorted task order
            task_order = self._topological_sort(tasks)
            
            self.log(f"Created {len(tasks)} tasks", "success")
            
            # Update state
            state["tasks"] = tasks
            state["task_order"] = task_order
            state["phase"] = "execution"
            
        except Exception as e:
            self.log(f"Error breaking down tasks: {e}", "error")
            state["errors"].append(f"Task Agent error: {str(e)}")
        
        return state
    
    def _topological_sort(self, tasks: List[Task]) -> List[str]:
        """Sort tasks based on dependencies (Kahn's algorithm)"""
        # Build adjacency list and in-degree count
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: 0 for t in tasks}
        graph = {t.id: [] for t in tasks}
        
        for task in tasks:
            for dep in task.dependencies:
                if dep in graph:
                    graph[dep].append(task.id)
                    in_degree[task.id] += 1
        
        # Start with tasks that have no dependencies
        queue = [tid for tid, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority (P0 first)
            queue.sort(key=lambda x: task_map[x].priority.value)
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Handle any remaining tasks (circular dependencies)
        remaining = [t.id for t in tasks if t.id not in result]
        result.extend(remaining)
        
        return result
