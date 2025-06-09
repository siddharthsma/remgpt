"""
Memory instructions block for LLM Context management.
"""

from typing import List, Optional
import logging
from ..base_block import BaseBlock
from ...types import Message, SystemMessage


class MemoryInstructionsBlock(BaseBlock):
    """Block containing memory instructions (read-only)."""
    
    def __init__(self, memory_content: str, name: str = "memory_instructions", logger: Optional[logging.Logger] = None):
        """Initialize with memory content."""
        super().__init__(name, is_read_only=True, logger=logger)
        
        # If no custom memory content is provided, use default context management instructions
        if not memory_content.strip():
            memory_content = self._get_default_context_instructions()
        
        self.memory_content = memory_content
    
    def _get_default_context_instructions(self) -> str:
        """Get default context management instructions for the LLM."""
        return """ENHANCED CONTEXT MANAGEMENT INSTRUCTIONS:

You have access to advanced context management functions that provide intelligent memory management. Here's when and how to use them:

1. TOPIC DRIFT DETECTED:
   - When you receive a message stating "TOPIC DRIFT DETECTED", the conversation has shifted to a new topic
   - You MUST immediately call 'save_current_topic' to summarize and save the previous conversation
   - The system may automatically recall similar topics from long-term memory
   - If a similar topic was recalled, acknowledge it and build upon the existing context

2. INTELLIGENT TOPIC MANAGEMENT FUNCTIONS:
   - save_current_topic(topic_summary, topic_key_facts): Save current conversation as new topic OR update existing similar topic
   - update_topic(topic_id, additional_summary, additional_key_facts): Explicitly update an existing topic with new information
   - recall_similar_topic(user_message): Search long-term memory for topics similar to user's message
   - evict_oldest_topic(): Remove oldest topic from working context when memory is full

3. SMART TOPIC UPDATING:
   - The system automatically detects when new conversations are similar to existing topics (similarity > 80%)
   - Instead of creating duplicate topics, similar conversations are merged intelligently
   - Embeddings and similarity scores determine when to update vs. create new topics
   - You'll see log messages indicating when topics are updated vs. created

4. TOPIC RECALL WORKFLOW:
   - When topic drift occurs, the system automatically searches for similar past topics
   - If found (similarity > 70%), relevant topics are loaded into your working context
   - You can also manually search using recall_similar_topic() for any user message
   - Recalled topics appear in your working context with their key facts

5. ENHANCED KEY FACTS DISPLAY:
   - Key facts are prominently displayed in working context with emoji formatting:
     📋 Topic: [Summary]
     🔑 Key Facts:
        1. [Fact 1]
        2. [Fact 2]
   - Facts are searchable and help with topic similarity detection
   - Updated topics merge key facts intelligently to avoid duplicates

6. FUNCTION CALLING GUIDELINES:
   * save_current_topic(topic_summary, topic_key_facts): REQUIRED when topic drift is detected
     - topic_summary: Concise summary of the conversation topic (1-2 sentences)
     - topic_key_facts: List of SPECIFIC, ACTIONABLE facts learned (CRITICAL!)
     
   * update_topic(topic_id, additional_summary, additional_key_facts): Update existing topic
     - topic_id: ID of the topic to update (from working context)
     - additional_summary: New information to merge with existing summary
     - additional_key_facts: New facts to add to existing facts
     
   * recall_similar_topic(user_message): Search for similar topics
     - user_message: The user's message to find similar topics for
     
   * evict_oldest_topic(): Free up memory when context is full

7. MEMORY EFFICIENCY:
   - Working context maintains optimal topics based on relevance and recency
   - Vector database stores long-term topic memory with semantic search
   - Automatic eviction prevents context overflow while preserving important information

📋 EXAMPLE USAGE:
save_current_topic(
  "Advanced microservices deployment strategies with Kubernetes", 
  [
    "Container orchestration with Kubernetes provides auto-scaling",
    "Service mesh (Istio) handles inter-service communication",
    "Blue-green deployment patterns minimize downtime",
    "User's company uses AWS EKS for managed Kubernetes",
    "Monitoring setup includes Prometheus and Grafana"
  ]
)

Remember: Your enhanced memory system learns from conversation patterns and automatically manages topic relationships. Be thorough with summaries and key facts for optimal memory performance."""
    
    def to_messages(self) -> List[Message]:
        """Convert to system message."""
        if not self.memory_content.strip():
            return []
        return [SystemMessage(content=f"Memory Instructions:\n{self.memory_content}")]
    
    def update_memory_content(self, new_content: str):
        """
        Update memory content (for dynamic instruction updates).
        
        Args:
            new_content: New memory instructions
        """
        self.memory_content = new_content
        self.logger.info("Memory instructions updated") 