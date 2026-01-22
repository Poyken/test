"""
AI Flow - Base Agent
Abstract base class for all AI agents
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import yaml
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from orchestrator.state import WorkflowState


class BaseAgent(ABC):
    """Base class for all AI agents in the workflow"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.llm = self._init_llm()
        self.json_parser = JsonOutputParser()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        config_file = Path(__file__).parent.parent / config_path
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _init_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the LLM based on configuration"""
        provider = self.config.get("llm", {}).get("provider", "gemini")
        
        if provider == "gemini":
            gemini_config = self.config.get("llm", {}).get("gemini", {})
            return ChatGoogleGenerativeAI(
                model=gemini_config.get("model", "gemini-2.0-flash-exp"),
                temperature=gemini_config.get("temperature", 0.1),
                max_output_tokens=gemini_config.get("max_tokens", 8192),
            )
        else:
            # Fallback to Gemini
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1,
            )
    
    def _get_agent_config(self) -> dict:
        """Get agent-specific configuration"""
        agent_name = self.__class__.__name__.lower().replace("agent", "_agent")
        return self.config.get("agents", {}).get(agent_name, {})
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for logging and identification"""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt that defines the agent's role and behavior"""
        pass
    
    @abstractmethod
    async def process(self, state: WorkflowState) -> WorkflowState:
        """
        Process the current state and return updated state.
        This is the main method that implements the agent's logic.
        """
        pass
    
    async def invoke_llm(
        self,
        user_prompt: str,
        output_schema: Optional[type[BaseModel]] = None,
    ) -> Any:
        """
        Invoke the LLM with the system prompt and user prompt.
        Optionally parse output to a Pydantic model.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        import asyncio
        import random
        
        max_retries = 10  # Tăng số lần retry cho Free Tier
        base_delay = 5    # Giây
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = await self.llm.ainvoke(messages)
                return self._parse_response(response, output_schema)
            except Exception as e:
                last_exception = e
                # Kiểm tra lỗi rate limit (429) hoặc quota
                error_str = str(e).lower()
                if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    if delay > 60: delay = 60 # Cap delay at 60s
                    self.log(f"Rate limit hit. Retrying in {delay:.2f}s (Attempt {attempt+1}/{max_retries})", level="warning")
                    await asyncio.sleep(delay)
                else:
                    raise e
        
        raise last_exception

    def _parse_response(self, response, output_schema):
        if not output_schema:
            return response.content

        # Parse JSON response to Pydantic model
        try:
            json_str = response.content
            # Extract JSON from markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            import json
            data = json.loads(json_str)
            
            if isinstance(data, list):
                return [output_schema(**item) for item in data]
            return output_schema(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {response.content}")
    
    def log(self, message: str, level: str = "info"):
        """Log a message with agent context"""
        from rich.console import Console
        console = Console()
        
        colors = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        color = colors.get(level, "white")
        console.print(f"[{color}][{self.name}][/{color}] {message}")
