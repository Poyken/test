"""
AI Flow - State Management
Shared state for the LangGraph workflow
"""

from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_NEEDED = "review_needed"


class TaskPriority(str, Enum):
    P0 = "P0"  # Critical
    P1 = "P1"  # High
    P2 = "P2"  # Medium
    P3 = "P3"  # Low


class TaskType(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    API = "api"
    TEST = "test"
    DOCUMENTATION = "documentation"


class UserStory(BaseModel):
    """User story extracted from meeting notes"""
    id: str
    title: str
    description: str
    role: str  # As a [role]
    feature: str  # I want [feature]
    benefit: str  # So that [benefit]
    acceptance_criteria: List[str]
    priority: TaskPriority
    complexity: Literal["S", "M", "L", "XL"]


class TechnicalSpec(BaseModel):
    """Technical specification for a feature"""
    feature_id: str
    database_changes: List[str]
    api_endpoints: List[str]
    frontend_components: List[str]
    dependencies: List[str]


class Task(BaseModel):
    """Atomic development task"""
    id: str
    title: str
    description: str
    type: TaskType
    story_id: str  # Reference to parent user story
    dependencies: List[str]  # Task IDs this depends on
    files_to_create: List[str]
    files_to_modify: List[str]
    estimated_minutes: int
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    generated_code: Optional[str] = None
    review_comments: Optional[List[str]] = None
    test_results: Optional[dict] = None
    retry_count: int = 0


class ReviewResult(BaseModel):
    """Code review result"""
    task_id: str
    approved: bool
    comments: List[str]
    suggested_fixes: List[str]
    security_issues: List[str]
    performance_issues: List[str]


class TestResult(BaseModel):
    """Test execution result"""
    task_id: str
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    coverage: float
    error_messages: List[str]


class WorkflowState(TypedDict):
    """
    Main state object for the LangGraph workflow.
    This state is passed between all agents and updated at each step.
    """
    # Input
    meeting_notes: str
    project_context: str  # Existing codebase context from RAG
    output_dir: str  # Directory for generated files
    
    # Phase 1: Requirements
    user_stories: List[UserStory]
    technical_specs: List[TechnicalSpec]
    
    # Phase 2: Planning
    tasks: List[Task]
    task_order: List[str]  # Ordered list of task IDs
    
    # Phase 3: Execution
    current_task_id: Optional[str]
    completed_tasks: List[str]
    failed_tasks: List[str]
    
    # Phase 4: Review & Testing
    review_results: List[ReviewResult]
    test_results: List[TestResult]
    
    # Output
    generated_files: dict  # {filepath: content}
    git_commits: List[str]
    
    # Metadata
    phase: Literal["requirements", "planning", "execution", "review", "testing", "deployment"]
    errors: List[str]
    warnings: List[str]
    started_at: str
    completed_at: Optional[str]


def create_initial_state(meeting_notes: str, project_context: str = "", output_dir: str = "./generated") -> WorkflowState:
    """Create initial workflow state from meeting notes"""
    return WorkflowState(
        meeting_notes=meeting_notes,
        project_context=project_context,
        output_dir=output_dir,
        user_stories=[],
        technical_specs=[],
        tasks=[],
        task_order=[],
        current_task_id=None,
        completed_tasks=[],
        failed_tasks=[],
        review_results=[],
        test_results=[],
        generated_files={},
        git_commits=[],
        phase="requirements",
        errors=[],
        warnings=[],
        started_at=datetime.now().isoformat(),
        completed_at=None,
    )
