"""
BaseAgent — shared foundation for all Claude-powered agents.
Each subclass declares its name, role, and implements process().
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic
import config


@dataclass
class AgentResult:
    """Standardised output returned by every agent."""
    agent: str
    success: bool
    data: Any
    message: str
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base for all enterprise agents."""

    name: str = "BaseAgent"
    description: str = "Abstract base agent"
    emoji: str = "🤖"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._activity_log: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, **kwargs) -> AgentResult:
        """Wrapper that times execution and catches errors."""
        t0 = time.time()
        self._log(f"Starting — {kwargs.get('description', '')}")
        try:
            result = self.process(**kwargs)
            result.duration_ms = (time.time() - t0) * 1000
            self._log(f"Completed in {result.duration_ms:.0f} ms")
            return result
        except Exception as exc:
            self.logger.exception("Agent %s failed", self.name)
            return AgentResult(
                agent=self.name,
                success=False,
                data=None,
                message=str(exc),
                duration_ms=(time.time() - t0) * 1000,
            )

    @abstractmethod
    def process(self, **kwargs) -> AgentResult:
        """Override in each agent subclass."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_claude(self, system: str, user_content: Any,
                     max_tokens: int = 2048) -> str:
        """Thin wrapper around the Claude messages API."""
        response = self._client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text

    def _log(self, msg: str) -> None:
        entry = f"[{self.name}] {msg}"
        self._activity_log.append(entry)
        self.logger.info(entry)

    @property
    def activity_log(self) -> list[str]:
        return list(self._activity_log)

    def status(self) -> dict:
        return {
            "agent": self.name,
            "description": self.description,
            "emoji": self.emoji,
            "log_entries": len(self._activity_log),
        }
