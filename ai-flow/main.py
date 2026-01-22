#!/usr/bin/env python3
"""
AI Flow - Main Entry Point
Run the AI-driven development workflow
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Configure Google AI
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from orchestrator.workflow import AIFlowOrchestrator
from rag.indexer import DocumentIndexer, CodebaseIndexer

app = typer.Typer(help="AI-Driven Development Flow")
console = Console()


@app.command()
def run(
    input_file: str = typer.Argument(
        ...,
        help="Path to meeting notes (text/markdown) or audio/video file"
    ),
    output_dir: str = typer.Option(
        "./generated",
        "--output", "-o",
        help="Directory to output generated code"
    ),
    docs_dir: str = typer.Option(
        None,
        "--docs", "-d",
        help="Path to project documentation directory"
    ),
    codebase_dir: str = typer.Option(
        None,
        "--codebase", "-c",
        help="Path to existing codebase for context"
    ),
    skip_rag: bool = typer.Option(
        False,
        "--skip-rag",
        help="Skip RAG indexing (faster but less context)"
    ),
):
    """
    Run the AI-driven development workflow.
    
    Examples:
        # From meeting notes
        python main.py meeting_notes.txt -o ./output
        
        # With project docs for context
        python main.py meeting.mp4 -d ./docs -o ./output
        
        # With existing codebase for patterns
        python main.py requirements.md -c ./src -o ./output
    """
    asyncio.run(_run_async(input_file, output_dir, docs_dir, codebase_dir, skip_rag))


async def _run_async(
    input_file: str,
    output_dir: str,
    docs_dir: str | None,
    codebase_dir: str | None,
    skip_rag: bool,
):
    """Async implementation of the run command"""
    
    console.print(Panel.fit(
        "[bold blue]AI-Driven Development Flow[/bold blue]\n"
        "Transforming ideas into code automatically",
        border_style="blue"
    ))
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("[red]Error: GOOGLE_API_KEY not set![/red]")
        console.print("Get your free API key at: https://makersuite.google.com/app/apikey")
        console.print("Then set it: export GOOGLE_API_KEY=your_key_here")
        raise typer.Exit(1)
    
    # Read input
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Determine if it's audio/video or text
    audio_video_extensions = {'.mp3', '.wav', '.m4a', '.mp4', '.webm', '.ogg'}
    
    if input_path.suffix.lower() in audio_video_extensions:
        meeting_notes = str(input_path.absolute())
        console.print(f"[yellow]Audio/Video file detected: {input_path.name}[/yellow]")
        console.print("[dim]Will transcribe using Gemini...[/dim]")
    else:
        meeting_notes = input_path.read_text(encoding='utf-8')
        console.print(f"[green]Text file loaded: {input_path.name}[/green]")
        console.print(f"[dim]{len(meeting_notes)} characters[/dim]")
    
    # Build context
    project_context = ""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Index docs if provided
        if docs_dir and not skip_rag:
            task = progress.add_task("Indexing documentation...", total=None)
            doc_indexer = DocumentIndexer()
            doc_count = doc_indexer.index_docs_directory(docs_dir)
            project_context += f"\n\n=== PROJECT DOCUMENTATION ===\n{doc_indexer.get_all_docs_content()}"
            progress.update(task, completed=True)
            console.print(f"[green]Indexed {doc_count} documentation files[/green]")
        
        # Index codebase if provided
        if codebase_dir and not skip_rag:
            task = progress.add_task("Indexing codebase...", total=None)
            code_indexer = CodebaseIndexer()
            code_count = code_indexer.index_directory(codebase_dir)
            
            # Get relevant context
            if code_count > 0:
                # Generate embeddings
                await code_indexer.generate_embeddings()
                
                # Get context for general code patterns
                project_context += f"\n\n=== EXISTING CODE PATTERNS ===\n"
                project_context += code_indexer.get_context_for_task(
                    "code structure, patterns, services, controllers, components",
                    top_k=10
                )
            
            progress.update(task, completed=True)
            console.print(f"[green]Indexed {code_count} code files[/green]")
    
    # Run the workflow
    console.print("\n[bold]Starting AI Flow...[/bold]\n")
    
    orchestrator = AIFlowOrchestrator()
    
    try:
        result = await orchestrator.run(
            meeting_notes=meeting_notes,
            project_context=project_context,
            output_dir=output_dir,
        )
        
        # Summary
        console.print(Panel(
            f"[green]✓ Workflow completed[/green]\n\n"
            f"[bold]User Stories:[/bold] {len(result.get('user_stories', []))}\n"
            f"[bold]Tasks:[/bold] {len(result.get('tasks', []))}\n"
            f"[bold]Completed:[/bold] {len(result.get('completed_tasks', []))}\n"
            f"[bold]Files Generated:[/bold] {len(result.get('generated_files', {}))}\n"
            f"[bold]Errors:[/bold] {len(result.get('errors', []))}\n\n"
            f"[dim]Output: {output_dir}[/dim]",
            title="Summary",
            border_style="green"
        ))
        
        # Show errors if any
        if result.get('errors'):
            console.print("\n[yellow]Errors encountered:[/yellow]")
            for error in result['errors']:
                console.print(f"  [red]• {error}[/red]")
        
        # Show generated files
        if result.get('generated_files'):
            console.print("\n[bold]Generated Files:[/bold]")
            for filepath in sorted(result['generated_files'].keys()):
                console.print(f"  [green]✓[/green] {filepath}")
        
    except Exception as e:
        console.print(f"[red]Error running workflow: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def demo():
    """Run a demo with the sample e-commerce docs"""
    docs_dir = Path(__file__).parent.parent / "docs"
    
    if not docs_dir.exists():
        console.print(f"[red]Docs directory not found: {docs_dir}[/red]")
        raise typer.Exit(1)
    
    console.print("[bold]Running demo with e-commerce project docs...[/bold]")
    
    asyncio.run(_run_async(
        input_file=str(docs_dir / "01-BRD.md"),
        output_dir="./generated",
        docs_dir=str(docs_dir),
        codebase_dir=None,
        skip_rag=False,
    ))


@app.command()
def version():
    """Show version information"""
    console.print("[bold]AI Flow[/bold] v1.0.0")
    console.print("AI-Driven Development Flow")
    console.print("\nModels: Gemini 2.0 Flash (Free Tier)")
    console.print("Framework: LangGraph + LangChain")


if __name__ == "__main__":
    app()
