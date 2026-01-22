"""
AI Flow - Transcriber Agent
Converts audio/video from meetings to text
Uses Gemini's multimodal capabilities (FREE)
"""

import base64
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from orchestrator.state import WorkflowState


class TranscriberAgent(BaseAgent):
    """
    Transcriber Agent
    
    Responsibilities:
    - Transcribe audio files (mp3, wav, m4a)
    - Transcribe video files (mp4, webm)
    - Extract text from meeting recordings
    - Summarize long transcripts
    
    Uses Gemini's multimodal free tier for transcription.
    """
    
    @property
    def name(self) -> str:
        return "Transcriber Agent"
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert meeting transcriber and summarizer.

Your role is to:
1. Accurately transcribe audio/video content
2. Identify different speakers when possible
3. Extract key decisions, action items, and requirements
4. Format the transcript in a clear, readable way

OUTPUT FORMAT:
Provide the transcript in the following format:

## Meeting Transcript

### Participants
- [Speaker 1]
- [Speaker 2]
...

### Transcript
[Timestamp if available] [Speaker]: [Content]
...

### Key Points
1. [Key decision or requirement]
2. ...

### Action Items
- [ ] [Action item with owner if mentioned]
- ...

### Technical Requirements Mentioned
- [Any specific technical requirements discussed]
- ..."""
    
    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio or video file"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        import google.generativeai as genai
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read and encode the file
        with open(path, "rb") as f:
            file_data = f.read()
        
        # Determine MIME type
        extension = path.suffix.lower()
        mime_types = {
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".m4a": "audio/m4a",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".ogg": "audio/ogg",
        }
        mime_type = mime_types.get(extension, "audio/mp3")
        
        self.log(f"Transcribing {path.name} ({mime_type})...")
        
        # Use Gemini's multimodal capability
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Upload file to Gemini
        uploaded_file = genai.upload_file(path, mime_type=mime_type)
        
        # Generate transcription
        response = model.generate_content([
            uploaded_file,
            "Transcribe this audio/video content. Include speaker identification if possible. "
            "Format the output as a meeting transcript with key points and action items."
        ])
        
        return response.text
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Process audio/video files if provided, otherwise pass through"""
        
        # Check if meeting_notes contains a file path
        meeting_input = state.get("meeting_notes", "")
        
        # Check if it's a file path
        if meeting_input.endswith(('.mp3', '.wav', '.m4a', '.mp4', '.webm', '.ogg')):
            try:
                self.log(f"Detected audio/video file: {meeting_input}")
                transcript = await self.transcribe_file(meeting_input)
                state["meeting_notes"] = transcript
                self.log("Transcription complete", "success")
            except Exception as e:
                self.log(f"Transcription error: {e}", "error")
                state["errors"].append(f"Transcriber Agent error: {str(e)}")
        else:
            # Already text, no transcription needed
            self.log("Input is text, skipping transcription")
        
        return state
    
    async def transcribe_and_summarize(self, file_path: str, max_length: int = 5000) -> str:
        """Transcribe and summarize long recordings"""
        transcript = await self.transcribe_file(file_path)
        
        if len(transcript) > max_length:
            self.log("Transcript is long, generating summary...")
            
            summary_prompt = f"""Summarize this meeting transcript, focusing on:
1. Key decisions made
2. Technical requirements discussed
3. Action items and owners
4. Timeline or deadlines mentioned

Keep the essential details for software development.

TRANSCRIPT:
{transcript}"""
            
            summary = await self.invoke_llm(summary_prompt)
            return f"## Summary\n{summary}\n\n## Full Transcript\n{transcript}"
        
        return transcript
