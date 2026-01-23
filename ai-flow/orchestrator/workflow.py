"""
AI Flow - Main Workflow Orchestrator
LangGraph-based workflow that coordinates all agents
"""

import asyncio
from datetime import datetime
from typing import Literal
from pathlib import Path

from langgraph.graph import StateGraph, END

from orchestrator.state import WorkflowState, create_initial_state, TaskStatus
from agents.transcriber_agent import TranscriberAgent
from agents.pm_agent import PMAgent
from agents.architect_agent import ArchitectAgent
from agents.task_agent import TaskAgent
from agents.code_agent import CodeAgent
from agents.review_agent import ReviewAgent
from agents.qa_agent import QAAgent
from agents.reflector_agent import ReflectorAgent


class AIFlowOrchestrator:
    """
    Main orchestrator for the AI-driven development workflow.
    Uses LangGraph to coordinate multiple AI agents.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        
        # Initialize agents
        self.transcriber = TranscriberAgent(config_path)
        self.pm_agent = PMAgent(config_path)
        self.architect_agent = ArchitectAgent(config_path)
        self.task_agent = TaskAgent(config_path)
        self.code_agent = CodeAgent(config_path)
        self.review_agent = ReviewAgent(config_path)
        self.qa_agent = QAAgent(config_path)
        self.reflector_agent = ReflectorAgent(config_path)
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph with our state type
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent/phase
        workflow.add_node("transcribe", self._transcribe_node)
        workflow.add_node("extract_requirements", self._requirements_node)
        workflow.add_node("design_architecture", self._architecture_node)
        workflow.add_node("plan_tasks", self._task_planning_node)
        workflow.add_node("execute_task", self._execute_task_node)
        workflow.add_node("reflector_check", self._reflector_node)
        workflow.add_node("review_code", self._review_node)
        workflow.add_node("generate_tests", self._test_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define the flow
        workflow.set_entry_point("transcribe")
        
        # Linear flow for planning phase
        workflow.add_edge("transcribe", "extract_requirements")
        workflow.add_edge("extract_requirements", "design_architecture")
        workflow.add_edge("design_architecture", "plan_tasks")
        
        # Conditional flow for execution phase
        workflow.add_conditional_edges(
            "plan_tasks",
            self._should_execute,
            {
                "execute": "execute_task",
                "done": "finalize",
            }
        )
        
        
        workflow.add_edge("execute_task", "reflector_check")
        
        workflow.add_conditional_edges(
            "reflector_check",
            self._check_reflector_feedback,
            {
                "revise": "execute_task",
                "approve": "review_code",
            }
        )
        
        workflow.add_conditional_edges(
            "review_code",
            self._should_regenerate,
            {
                "regenerate": "execute_task",
                "continue": "generate_tests",
            }
        )
        
        workflow.add_conditional_edges(
            "generate_tests",
            self._has_more_tasks,
            {
                "more_tasks": "execute_task",
                "done": "finalize",
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # Node implementations
    async def _transcribe_node(self, state: WorkflowState) -> WorkflowState:
        """Transcribe audio/video if needed"""
        return await self.transcriber.process(state)
    
    async def _requirements_node(self, state: WorkflowState) -> WorkflowState:
        """Extract requirements from meeting notes"""
        state["phase"] = "requirements"
        return await self.pm_agent.process(state)
    
    async def _architecture_node(self, state: WorkflowState) -> WorkflowState:
        """Design technical architecture"""
        return await self.architect_agent.process(state)
    
    async def _task_planning_node(self, state: WorkflowState) -> WorkflowState:
        """Plan and prioritize tasks"""
        state["phase"] = "planning"
        return await self.task_agent.process(state)
    
    async def _execute_task_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the next task (generate code)"""
        state["phase"] = "execution"
        
        # Find next pending task
        for task_id in state["task_order"]:
            task = next((t for t in state["tasks"] if t.id == task_id), None)
            if task and task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.FAILED]:
                # Check if dependencies are met
                deps_met = all(
                    any(t.id == dep and t.status == TaskStatus.COMPLETED 
                        for t in state["tasks"])
                    for dep in task.dependencies
                )
                if deps_met or not task.dependencies:
                    state["current_task_id"] = task_id
                    task.status = TaskStatus.IN_PROGRESS
                    break
        
        return await self.code_agent.process(state)
    
    async def _reflector_node(self, state: WorkflowState) -> WorkflowState:
        """Critique generated code"""
        return await self.reflector_agent.process(state)

    async def _review_node(self, state: WorkflowState) -> WorkflowState:
        """Review generated code"""
        state["phase"] = "review"
        return await self.review_agent.process(state)
    
    async def _test_node(self, state: WorkflowState) -> WorkflowState:
        """Generate tests for the task"""
        state["phase"] = "testing"
        return await self.qa_agent.process(state)
    
    async def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow"""
        state["phase"] = "deployment"
        state["completed_at"] = datetime.now().isoformat()
        
        # Summary
        completed = len(state["completed_tasks"])
        failed = len(state["failed_tasks"])
        total = len(state["tasks"])
        
        print(f"\n{'='*60}")
        print(f"AI FLOW COMPLETED")
        print(f"{'='*60}")
        print(f"Tasks: {completed}/{total} completed, {failed} failed")
        print(f"Files generated: {len(state['generated_files'])}")
        print(f"Errors: {len(state['errors'])}")
        print(f"Duration: {state['started_at']} → {state['completed_at']}")
        print(f"{'='*60}\n")
        
        return state
    
    # Conditional edge functions
    def _should_execute(self, state: WorkflowState) -> Literal["execute", "done"]:
        """Determine if there are tasks to execute"""
        if not state.get("tasks"):
            return "done"
        
        pending = [t for t in state["tasks"] if t.status == TaskStatus.PENDING]
        return "execute" if pending else "done"
    
    def _should_regenerate(self, state: WorkflowState) -> Literal["regenerate", "continue"]:
        """Determine if code needs to be regenerated after review"""
        if not state.get("review_results"):
            return "continue"
        
        current_task_id = state.get("current_task_id")
        current_task = next((t for t in state["tasks"] if t.id == current_task_id), None)
        
        if current_task and current_task.status == TaskStatus.FAILED:
            if current_task.retry_count < 3:
                current_task.status = TaskStatus.PENDING
                current_task.retry_count += 1
                return "regenerate"
        
        return "continue"
    
    def _has_more_tasks(self, state: WorkflowState) -> Literal["more_tasks", "done"]:
        """Check if there are more tasks to process"""
        pending = [t for t in state["tasks"] if t.status == TaskStatus.PENDING]
        return "more_tasks" if pending else "done"

    def _check_reflector_feedback(self, state: WorkflowState) -> Literal["revise", "approve"]:
        """Determine if code needs revision based on Reflector feedback"""
        feedback = state.get("reflector_feedback")
        if feedback and feedback.get("status") == "needs_revision":
            return "revise"
        return "approve"
    
    async def run(
        self,
        meeting_notes: str,
        project_context: str = "",
        output_dir: str = "./generated",
    ) -> WorkflowState:
        """
        Run the complete AI-driven development workflow.
        
        Args:
            meeting_notes: Meeting notes text or path to audio/video file
            project_context: Existing codebase context (from RAG)
            output_dir: Directory to write generated files
        
        Returns:
            Final workflow state with all generated artifacts
        """
        # Create initial state
        state = create_initial_state(meeting_notes, project_context, output_dir)
        
        print(f"\n{'='*60}")
        print("AI FLOW STARTED")
        print(f"{'='*60}")
        print(f"Input: {meeting_notes[:100]}...")
        print(f"Output dir: {output_dir}")
        print(f"{'='*60}\n")
        
        # Run the workflow
        final_state = await self.graph.ainvoke(state)
        
        # Write generated files
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for filepath, content in final_state.get("generated_files", {}).items():
            file_path = output_path / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            print(f"Written: {file_path}")
        
        return final_state


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Driven Development Flow")
    parser.add_argument("input", help="Meeting notes file or text")
    parser.add_argument("--output", "-o", default="./generated", help="Output directory")
    parser.add_argument("--context", "-c", default="", help="Project context file")
    args = parser.parse_args()
    
    # Read input
    input_path = Path(args.input)
    if input_path.exists() and input_path.suffix in ['.txt', '.md']:
        meeting_notes = input_path.read_text()
    else:
        meeting_notes = args.input
    
    # Read context if provided
    context = ""
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            context = context_path.read_text()
    
    # Run the flow
    orchestrator = AIFlowOrchestrator()
    result = await orchestrator.run(
        meeting_notes=meeting_notes,
        project_context=context,
        output_dir=args.output,
    )
    
    print(f"\nWorkflow completed with {len(result['errors'])} errors")


if __name__ == "__main__":
    asyncio.run(main())
