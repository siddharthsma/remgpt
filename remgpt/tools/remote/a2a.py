"""
Agent-to-Agent (A2A) protocol client implementation.
"""

from typing import Dict, Any, List, Optional, Literal
import uuid

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base import RemoteToolProtocol


class TaskRequestBuilder:
    """Builder for A2A task requests."""
    
    @staticmethod
    def build_send_task_params(
        *,
        task_id: str,
        role: Literal["user", "agent"] = "user",
        text: str,
        data: Dict[str, Any] = None,
        file_uri: str = None,
        session_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Build parameters for a send task request."""
        parts = []
        
        if text:
            parts.append({"type": "text", "text": text})
        
        if data:
            parts.append({"type": "data", "data": data})
        
        if file_uri:
            parts.append({
                "type": "file", 
                "file": {"uri": file_uri}
            })
        
        return {
            "id": task_id,
            "sessionId": session_id or str(uuid.uuid4()),
            "message": {
                "role": role,
                "parts": parts
            },
            "metadata": metadata
        }


class A2AProtocol(RemoteToolProtocol):
    """Agent-to-Agent protocol client using JSON-RPC task-based communication."""
    
    def __init__(self, agent_url: str, agent_name: Optional[str] = None):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx package required for A2A protocol. Install with: pip install httpx")
        
        self.agent_url = agent_url
        self.agent_name = agent_name or f"agent_{agent_url.split('/')[-1]}"
        self.request_builder = TaskRequestBuilder()
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available A2A tools."""
        return [{
            "name": "send_task",
            "description": f"Send a task to the {self.agent_name} agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string", 
                        "description": "Message text to send to the agent"
                    },
                    "data": {
                        "type": "object", 
                        "description": "Optional structured data to include with the task"
                    },
                    "role": {
                        "type": "string", 
                        "enum": ["user", "agent"], 
                        "default": "user",
                        "description": "Role of the sender"
                    },
                    "file_uri": {
                        "type": "string",
                        "description": "Optional URI of a file to include with the task"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID for task continuity"
                    }
                },
                "required": ["text"]
            }
        }]
    
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Call an A2A tool."""
        if tool_name != "send_task":
            raise ValueError(f"Unknown A2A tool: {tool_name}")
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Build task parameters using the builder
        task_params = self.request_builder.build_send_task_params(
            task_id=task_id,
            role=args.get("role", "user"),
            text=args["text"],
            data=args.get("data"),
            file_uri=args.get("file_uri"),
            session_id=args.get("session_id"),
            metadata=args.get("metadata")
        )
        
        # Create JSON-RPC request
        request_payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": task_params,
            "id": str(uuid.uuid4())
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(self.agent_url, json=request_payload)
                response.raise_for_status()
                result = response.json()
                
                # Handle JSON-RPC response format
                if "result" in result:
                    task_result = result["result"]
                    
                    # Extract useful information from the task
                    if isinstance(task_result, dict):
                        return {
                            "task_id": task_result.get("id", task_id),
                            "status": task_result.get("status", {}).get("state", "submitted"),
                            "session_id": task_result.get("sessionId"),
                            "response": task_result
                        }
                    return {"task_id": task_id, "response": task_result}
                
                elif "error" in result:
                    error_info = result["error"]
                    error_msg = f"A2A error ({error_info.get('code', 'unknown')}): {error_info.get('message', 'Unknown error')}"
                    raise Exception(error_msg)
                
                else:
                    return {"task_id": task_id, "response": result}
                    
            except Exception as e:
                # Handle both httpx specific exceptions and general exceptions
                if "HTTPStatusError" in str(type(e)):
                    raise Exception(f"A2A HTTP error: {getattr(e, 'response', {}).get('status_code', 'unknown')}")
                elif "RequestError" in str(type(e)):
                    raise Exception(f"A2A request error: {e}")
                else:
                    # Re-raise the original exception for other cases (like our mock tests)
                    raise e
    
    async def send_task(
        self,
        text: str,
        role: Literal["user", "agent"] = "user",
        data: Dict[str, Any] = None,
        file_uri: str = None,
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Direct method to send a task to the agent.
        This is the primary method that gets exposed as a tool.
        """
        return await self.call_tool("send_task", {
            "text": text,
            "role": role,
            "data": data,
            "file_uri": file_uri,
            "session_id": session_id,
            "metadata": metadata
        }) 