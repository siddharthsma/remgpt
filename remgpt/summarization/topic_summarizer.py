"""
Topic summarizer for context management.
"""

import time
import logging
from typing import List, Optional, Callable
import numpy as np

from ..core.types import Message, SystemMessage
from .topic import Topic


class TopicSummarizer:
    """
    Summarizes a collection of messages into a Topic.
    
    This class can use different summarization strategies:
    - LLM-based summarization (if LLM client available)
    - Extractive summarization (using key sentences)
    - Template-based summarization (fallback)
    """
    
    def __init__(
        self,
        llm_client: Optional[Callable] = None,
        max_summary_length: int = 200,
        max_key_facts: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize topic summarizer.
        
        Args:
            llm_client: Optional LLM client for advanced summarization
            max_summary_length: Maximum length of summary
            max_key_facts: Maximum number of key facts to extract
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.max_summary_length = max_summary_length
        self.max_key_facts = max_key_facts
        self.logger = logger or logging.getLogger(__name__)
    
    async def summarize_messages(
        self,
        messages: List[Message],
        mean_embedding: np.ndarray,
        topic_id: Optional[str] = None
    ) -> Topic:
        """
        Summarize a list of messages into a Topic.
        
        Args:
            messages: Messages to summarize
            mean_embedding: Mean embedding of the messages
            topic_id: Optional topic ID, will generate if not provided
            
        Returns:
            Topic object containing summary and metadata
        """
        if not messages:
            raise ValueError("Cannot summarize empty message list")
        
        # Generate topic ID if not provided
        if topic_id is None:
            topic_id = f"topic_{int(time.time())}"
        
        self.logger.info(f"Summarizing {len(messages)} messages into topic {topic_id}")
        
        # Extract text content from messages
        message_texts = []
        for msg in messages:
            if isinstance(msg.content, str):
                message_texts.append(msg.content)
            elif isinstance(msg.content, list):
                # Extract text from structured content
                text_parts = []
                for item in msg.content:
                    if hasattr(item, 'text'):
                        text_parts.append(item.text)
                    elif isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                message_texts.append(" ".join(text_parts))
            else:
                message_texts.append(str(msg.content) if msg.content else "")
        
        # Try LLM-based summarization first
        if self.llm_client:
            try:
                summary, key_facts = await self._llm_summarize(message_texts)
            except Exception as e:
                self.logger.warning(f"LLM summarization failed: {e}, falling back to extractive")
                summary, key_facts = self._extractive_summarize(message_texts)
        else:
            summary, key_facts = self._extractive_summarize(message_texts)
        
        # Create topic
        topic = Topic(
            id=topic_id,
            summary=summary,
            key_facts=key_facts,
            mean_embedding=mean_embedding,
            original_messages=messages.copy(),
            timestamp=time.time(),
            message_count=len(messages),
            metadata={
                "creation_method": "llm" if self.llm_client else "extractive",
                "original_length": sum(len(text) for text in message_texts)
            }
        )
        
        self.logger.info(f"Created topic {topic_id}: {summary[:100]}...")
        return topic
    
    async def _llm_summarize(self, message_texts: List[str]) -> tuple[str, List[str]]:
        """
        Use LLM to summarize messages.
        
        Args:
            message_texts: List of message texts to summarize
            
        Returns:
            Tuple of (summary, key_facts)
        """
        # Combine all messages
        combined_text = "\n".join([f"Message {i+1}: {text}" for i, text in enumerate(message_texts)])
        
        # Create summarization prompt
        prompt = f"""Please summarize the following conversation messages, maintaining key facts and important details:

{combined_text}

Provide:
1. A concise summary (max {self.max_summary_length} characters)
2. Up to {self.max_key_facts} key facts or important details

Format your response as:
SUMMARY: [your summary here]
KEY_FACTS:
- [fact 1]
- [fact 2]
- [etc]"""
        
        # Create system message for LLM
        summary_messages = [SystemMessage(content=prompt)]
        
        # Call LLM
        response = await self.llm_client(summary_messages)
        
        # Parse response
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        # Extract summary and key facts
        summary, key_facts = self._parse_llm_response(response_text)
        
        return summary, key_facts
    
    def _parse_llm_response(self, response_text: str) -> tuple[str, List[str]]:
        """Parse LLM response to extract summary and key facts."""
        lines = response_text.strip().split('\n')
        
        summary = ""
        key_facts = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                current_section = "summary"
            elif line.startswith("KEY_FACTS:"):
                current_section = "facts"
            elif line.startswith("- ") and current_section == "facts":
                fact = line.replace("- ", "").strip()
                if fact and len(key_facts) < self.max_key_facts:
                    key_facts.append(fact)
            elif current_section == "summary" and not summary:
                summary = line
        
        # Fallback if parsing failed
        if not summary:
            summary = response_text[:self.max_summary_length].strip()
        
        if not key_facts:
            # Try to extract sentences as key facts
            sentences = response_text.split('. ')
            key_facts = sentences[:self.max_key_facts]
        
        # Truncate summary if too long
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length-3] + "..."
        
        return summary, key_facts
    
    def _extractive_summarize(self, message_texts: List[str]) -> tuple[str, List[str]]:
        """
        Extractive summarization using simple heuristics.
        
        Args:
            message_texts: List of message texts to summarize
            
        Returns:
            Tuple of (summary, key_facts)
        """
        all_text = " ".join(message_texts)
        
        # Simple extractive summary: take first and last sentences
        sentences = [s.strip() for s in all_text.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            summary = all_text[:self.max_summary_length]
        else:
            # Take first sentence and last sentence
            summary = f"{sentences[0]}. {sentences[-1]}."
        
        # Truncate if too long
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length-3] + "..."
        
        # Extract key facts as important sentences (longer sentences often contain more info)
        important_sentences = sorted(sentences, key=len, reverse=True)[:self.max_key_facts]
        key_facts = [s[:100] + "..." if len(s) > 100 else s for s in important_sentences]
        
        return summary, key_facts 