"""
AI Flow - Code Generation Agent
Generates code for various task types
"""

import json
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from orchestrator.state import WorkflowState, Task, TaskStatus, TaskType


class CodeAgent(BaseAgent):
    """
    Code Generation Agent
    
    Responsibilities:
    - Generate production-quality code for tasks
    - Follow tech stack conventions (NestJS, Next.js, Prisma)
    - Include type annotations
    - Add proper error handling
    - Write inline documentation
    """
    
    @property
    def name(self) -> str:
        return "Code Agent"
    
    @property
    def system_prompt(self) -> str:
        return """You are a Senior Full-Stack Developer with expertise in:
- NestJS (backend), Next.js (frontend), TypeScript
- Prisma ORM, PostgreSQL
- Multi-tenant SaaS architecture

CODE QUALITY REQUIREMENTS:
1. TypeScript strict mode - no implicit any
2. Proper error handling with try/catch
3. Input validation using class-validator (backend) or Zod (frontend)
4. JSDoc comments for all public functions
5. Follow existing code patterns from context

BACKEND (NestJS) PATTERNS:
- Controllers handle HTTP, services handle business logic
- DTOs for request/response validation
- Guards for authentication/authorization
- Inject tenantId from AsyncLocalStorage via tenant service

FRONTEND (Next.js) PATTERNS:
- Server Components by default, 'use client' only when needed
- Server Actions for mutations
- Zustand for client state
- TailwindCSS for styling

OUTPUT FORMAT:
Respond with a JSON object containing a "files" array.
Each file must have:
- path: string (relative file path)
- content: string (complete file content)
- action: string ("create" or "modify")

For modifications, include the COMPLETE new file content, not just changes.

Example:
```json
{
  "files": [
    {
      "path": "src/users/users.service.ts",
      "content": "import { Injectable } from '@nestjs/common';\\n\\n@Injectable()\\nexport class UsersService {\\n  // ...\\n}",
      "action": "create"
    }
  ]
}
```"""
    
    def _get_task_specific_prompt(self, task: Task) -> str:
        """Get additional context based on task type"""
        prompts = {
            TaskType.DATABASE: """
DATABASE TASK GUIDELINES:
- Use Prisma schema syntax
- Include proper indexes
- Add tenantId to tenant-scoped models
- Include createdAt, updatedAt, deletedAt
- Use proper relations with onDelete behavior
""",
            TaskType.BACKEND: """
BACKEND TASK GUIDELINES:
- Create NestJS service with dependency injection
- Use async/await for all database operations
- Include proper error handling (throw HttpException)
- Add logging for important operations
- Follow repository pattern with Prisma
""",
            TaskType.API: """
API TASK GUIDELINES:
- Create NestJS controller with proper decorators
- Use DTOs for request validation
- Return consistent response format
- Add Swagger documentation decorators
- Include proper HTTP status codes
""",
            TaskType.FRONTEND: """
FRONTEND TASK GUIDELINES:
- Use Next.js 14+ App Router patterns
- Server Components by default
- Add 'use client' only when needed (hooks, interactivity)
- Use TailwindCSS for styling
- Implement loading and error states
""",
            TaskType.TEST: """
TEST TASK GUIDELINES:
- Use Jest for unit tests
- Mock external dependencies
- Test happy path and error cases
- Include edge cases
- Aim for high coverage on critical paths
""",
        }
        return prompts.get(task.type, "")
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Generate code for the current task"""
        current_task_id = state.get("current_task_id")
        
        if not current_task_id:
            self.log("No current task to process", "warning")
            return state
        
        # Find the current task
        task = None
        for t in state["tasks"]:
            if t.id == current_task_id:
                task = t
                break
        
        if not task:
            self.log(f"Task {current_task_id} not found", "error")
            return state
        
        self.log(f"Generating code for task: {task.title}")
        
        # Build context from existing files
        existing_files_context = ""
        if state.get("generated_files"):
            existing_files_context = "\n\nEXISTING GENERATED FILES:\n"
            for path, content in state["generated_files"].items():
                existing_files_context += f"\n--- {path} ---\n{content[:2000]}...\n"
        
        # Build context from project
        project_context = state.get("project_context", "")
        
        user_prompt = f"""Generate code for the following task.

TASK: {task.id}
Title: {task.title}
Description: {task.description}
Type: {task.type.value}
Files to create: {', '.join(task.files_to_create) if task.files_to_create else 'Determine based on task'}
Files to modify: {', '.join(task.files_to_modify) if task.files_to_modify else 'None'}

{self._get_task_specific_prompt(task)}

PROJECT CONTEXT:
{project_context}
{existing_files_context}

Generate complete, production-ready code.
Respond with ONLY valid JSON, no other text."""

        try:
            response = await self.invoke_llm(user_prompt)
            
            # Parse the response
            response_text = response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(response_text)
            files_data = data.get("files", [])
            
            # Store generated code in state
            for file_info in files_data:
                path = file_info["path"]
                content = file_info["content"]
                state["generated_files"][path] = content
                self.log(f"Generated: {path}", "success")
            
            # Update task status
            for t in state["tasks"]:
                if t.id == current_task_id:
                    t.status = TaskStatus.REVIEW_NEEDED
                    t.generated_code = json.dumps(files_data)
                    break
            
            self.log(f"Generated {len(files_data)} files for task {task.id}", "success")
            
        except Exception as e:
            self.log(f"Error generating code: {e}", "error")
            state["errors"].append(f"Code Agent error for {task.id}: {str(e)}")
            
            # Mark task as failed
            for t in state["tasks"]:
                if t.id == current_task_id:
                    t.status = TaskStatus.FAILED
                    t.retry_count += 1
                    break
        
        return state
