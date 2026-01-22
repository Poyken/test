"""
AI Flow - Review Agent
Reviews generated code for quality and issues
"""

import json
from typing import List

from agents.base_agent import BaseAgent
from orchestrator.state import WorkflowState, Task, TaskStatus, ReviewResult


class ReviewAgent(BaseAgent):
    """
    Code Review Agent
    
    Responsibilities:
    - Review generated code for quality
    - Check for security vulnerabilities
    - Verify best practices are followed
    - Suggest improvements
    - Approve or request changes
    """
    
    @property
    def name(self) -> str:
        return "Review Agent"
    
    @property
    def system_prompt(self) -> str:
        return """You are a Senior Code Reviewer with expertise in security and best practices.

REVIEW CHECKLIST:
1. **Type Safety**: No implicit any, proper TypeScript types
2. **Error Handling**: Proper try/catch, error responses
3. **Security**:
   - Input validation present
   - No SQL injection (using ORM properly)
   - No XSS vulnerabilities
   - Tenant isolation (tenantId checks)
   - No hardcoded secrets
4. **Performance**:
   - No N+1 queries
   - Proper indexing considered
   - No memory leaks
5. **Best Practices**:
   - Single responsibility
   - Proper separation of concerns
   - Code is testable
   - Documentation/comments present

REVIEW SEVERITY:
- CRITICAL: Must fix before merge (security, data loss risk)
- HIGH: Should fix (bugs, performance issues)
- MEDIUM: Recommended (code quality, maintainability)
- LOW: Nice to have (style, minor improvements)

OUTPUT FORMAT:
Respond with a JSON object:
```json
{
  "approved": boolean,
  "comments": ["General review comments"],
  "suggested_fixes": ["Specific code changes needed"],
  "security_issues": ["Any security concerns"],
  "performance_issues": ["Any performance concerns"]
}
```

If approved is false, suggested_fixes MUST contain actionable items."""
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Review code for the current task"""
        current_task_id = state.get("current_task_id")
        
        if not current_task_id:
            self.log("No current task to review", "warning")
            return state
        
        # Find the current task
        task = None
        for t in state["tasks"]:
            if t.id == current_task_id:
                task = t
                break
        
        if not task or not task.generated_code:
            self.log(f"No code to review for task {current_task_id}", "warning")
            return state
        
        self.log(f"Reviewing code for task: {task.title}")
        
        # Get the generated files
        try:
            files_data = json.loads(task.generated_code)
        except:
            files_data = []
        
        # Format files for review
        files_text = ""
        for file_info in files_data:
            files_text += f"\n--- {file_info['path']} ---\n{file_info['content']}\n"
        
        user_prompt = f"""Review the following code generated for this task.

TASK: {task.id}
Title: {task.title}
Type: {task.type.value}
Description: {task.description}

GENERATED CODE:
{files_text}

Review the code thoroughly and provide your assessment.
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
            
            # Create review result
            review = ReviewResult(
                task_id=task.id,
                approved=data.get("approved", False),
                comments=data.get("comments", []),
                suggested_fixes=data.get("suggested_fixes", []),
                security_issues=data.get("security_issues", []),
                performance_issues=data.get("performance_issues", []),
            )
            
            state["review_results"].append(review)
            
            # Update task status
            for t in state["tasks"]:
                if t.id == current_task_id:
                    if review.approved:
                        t.status = TaskStatus.COMPLETED
                        state["completed_tasks"].append(t.id)
                        self.log(f"Task {task.id} APPROVED", "success")
                    else:
                        t.status = TaskStatus.FAILED
                        t.review_comments = review.comments + review.suggested_fixes
                        self.log(f"Task {task.id} needs changes: {len(review.suggested_fixes)} fixes required", "warning")
                    break
            
        except Exception as e:
            self.log(f"Error reviewing code: {e}", "error")
            state["errors"].append(f"Review Agent error for {task.id}: {str(e)}")
        
        return state
