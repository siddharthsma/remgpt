"""
FastAPI application for RemGPT with streaming message processing.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from .orchestration import ConversationOrchestrator, create_orchestrator
from .llm import Event
from .context import create_context_manager
from .types import Message, UserMessage, MessageRole
from .config import get_config


# Error Handling
# =============

class RemGPTError(BaseModel):
    """Standard error response model."""
    error: str
    message: str
    request_id: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None


# AUTHENTICATION ASSUMPTIONS:
# =================================
# This implementation assumes that authentication will eventually be handled via:
# 1. OAuth 2.0 / OpenID Connect providers (Google, Auth0, etc.)
# 2. JWT tokens containing user identity and claims
# 3. Standard HTTP Authorization headers: "Bearer <jwt_token>"
# 
# Current implementation provides a foundation that can be easily extended
# to integrate with OAuth providers without changing the core message processing logic.
#
# The separation of auth (headers) from message content (request body) follows
# HTTP standards and makes OAuth integration straightforward.


# Request/Response Models
# NOTE: Removed MessageRequest - using UserMessage directly to eliminate duplication
# This ensures consistency with the core type system and supports rich content

class ContextConfig(BaseModel):
    """Configuration for context initialization."""
    max_tokens: int = Field(default=4000, description="Maximum token limit")
    system_instructions: str = Field(default="", description="System instructions")
    memory_content: str = Field(default="", description="Memory content")
    tools: list = Field(default_factory=list, description="Tool definitions")
    
    # Remote tool configuration
    mcp_servers: List[str] = Field(default_factory=list, description="MCP server URLs or paths")
    a2a_agents: List[str] = Field(default_factory=list, description="A2A agent base URLs")
    
    # Note: Model is no longer needed here since token counting automatically
    # adapts to the LLM client used by the orchestrator


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    orchestrator_status: str
    queue_size: int
    context_summary: Dict[str, Any]
    registered_tools: list


# Authentication Dependencies
# ===========================

async def get_current_user(
    authorization: str = Header(None, description="Bearer token for user authentication")
) -> str:
    """
    Extract and verify the current user from HTTP Authorization header.
    
    ASSUMPTIONS for OAuth/OpenID integration:
    - Authorization header contains: "Bearer <jwt_token>"
    - JWT token contains user identity claims (sub, email, name, etc.)
    - Token validation will be done against OAuth provider's public keys
    - User identity will be extracted from verified JWT claims
    
    Current implementation is a placeholder for development/testing.
    
    Args:
        authorization: HTTP Authorization header with Bearer token
        
    Returns:
        Verified user identifier (extracted from JWT claims)
        
    Raises:
        HTTPException: If authentication fails
        
    TODO: Replace with actual OAuth/JWT verification:
        - Verify JWT signature using OAuth provider's public keys
        - Check token expiration and issuer
        - Extract user claims (sub, email, name, etc.)
        - Use claims like 'sub', 'email', or 'preferred_username' as user ID
        - Optionally check user permissions/roles from claims
    
    SECURITY NOTE:
    - User identity comes ONLY from verified JWT token claims
    - Never trust additional headers for user identification
    - JWT should be the single source of truth for user identity
    """
    # Development/testing mode - accept any authorization header
    if not authorization:
        raise HTTPException(
            status_code=401, 
            detail="Missing Authorization header. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    # PLACEHOLDER: In production, this would be JWT verification
    if not token or len(token) < 10:  # Basic validation for development
        raise HTTPException(
            status_code=401,
            detail="Invalid or malformed token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # PLACEHOLDER: Extract user from JWT token
    # In OAuth implementation, this would:
    # 1. Verify JWT signature against OAuth provider's public keys
    # 2. Check token expiration (exp claim)
    # 3. Verify issuer (iss claim) 
    # 4. Extract user identity from claims:
    #    - user_id = jwt_payload.get('sub')  # Subject claim
    #    - or jwt_payload.get('email')       # Email claim
    #    - or jwt_payload.get('preferred_username')  # Username claim
    
    # Mock user extraction for development
    user_id = f"user_{token[:8]}"  # Based on token for consistency
    
    logger.info(f"Authenticated user: {user_id}")
    return user_id


# Global queue for incoming messages
message_queue: asyncio.Queue = None
orchestrator: ConversationOrchestrator = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global message_queue, orchestrator
    
    # Startup
    logger.info("Starting RemGPT API...")
    message_queue = asyncio.Queue()
    
    # Initialize with default context manager
    context_manager = create_context_manager(
        max_tokens=4000,
        system_instructions="You are a helpful AI assistant."
    )
    
    # Initialize orchestrator with remote tool support
    orchestrator = await create_orchestrator(
        context_manager=context_manager,
        auto_initialize_remote_tools=False  # No remote tools in default setup
    )
    
    # Start background task to process queue
    task = asyncio.create_task(process_message_queue())
    
    logger.info("RemGPT API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RemGPT API...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# Create FastAPI app
app = FastAPI(
    title="RemGPT API",
    description="Streaming conversational AI API with context management and OAuth-ready authentication",
    version="0.1.0",
    lifespan=lifespan
)


# Global Exception Handlers
# =========================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    request_id = str(uuid.uuid4())
    logger.error(f"Validation error for request {request_id}: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content=RemGPTError(
            error="VALIDATION_ERROR",
            message="Invalid request data",
            request_id=request_id,
            timestamp=time.time(),
            details={"validation_errors": exc.errors()}
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    request_id = str(uuid.uuid4())
    
    return JSONResponse(
        status_code=exc.status_code,
        content=RemGPTError(
            error="HTTP_ERROR",
            message=exc.detail,
            request_id=request_id,
            timestamp=time.time(),
            details={"status_code": exc.status_code}
        ).model_dump(),
        headers=getattr(exc, 'headers', None)
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception for request {request_id}: {exc}", exc_info=True)
    
    # Don't expose internal error details in production
    config = get_config()
    details = {"exception_type": type(exc).__name__}
    if config.debug:
        details["exception_message"] = str(exc)
    
    return JSONResponse(
        status_code=500,
        content=RemGPTError(
            error="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred",
            request_id=request_id,
            timestamp=time.time(),
            details=details
        ).model_dump()
    )


async def process_message_queue():
    """
    Background task to process the message queue.
    
    This function runs continuously, processing messages one at a time
    and routing events back to the correct client response queue.
    
    The sender information is preserved from authentication and included
    in processing for audit trails and user-specific behavior.
    """
    global message_queue, orchestrator
    
    while True:
        try:
            # Get message from queue
            queue_item = await message_queue.get()
            message = queue_item["message"]
            response_queue = queue_item["response_queue"]
            
            # Log message processing with authenticated sender
            logger.info(f"Processing message from user '{message.name}': {message.role.value}")
            
            # Process message through orchestrator
            async for event in orchestrator.process_message(message):
                # Add sender context to events for audit and personalization
                if hasattr(event, 'data') and isinstance(event.data, dict):
                    event.data["sender"] = message.name
                
                await response_queue.put(event)
            
            # Signal completion
            await response_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in message processing: {e}")
            # Put error event in response queue if available
            if 'response_queue' in locals():
                error_event = Event(
                    type="error",
                    data={"error": str(e), "error_type": type(e).__name__},
                    timestamp=asyncio.get_event_loop().time()
                )
                await response_queue.put(error_event)
                await response_queue.put(None)


async def format_sse(event: Event) -> str:
    """Format event as Server-Sent Event."""
    data = {
        "type": event.type,
        "data": event.data,
        "timestamp": event.timestamp
    }
    return f"data: {json.dumps(data)}\n\n"


async def stream_response_generator(response_queue: asyncio.Queue):
    """Generate SSE formatted responses from the response queue."""
    try:
        while True:
            event = await response_queue.get()
            
            if event is None:  # Signal for completion
                yield "data: {\"type\": \"stream_end\", \"data\": {}, \"timestamp\": " + str(asyncio.get_event_loop().time()) + "}\n\n"
                break
            
            yield await format_sse(event)
            
    except Exception as e:
        logger.error(f"Error in stream generator: {e}")
        error_data = {
            "type": "stream_error", 
            "data": {"error": str(e)}, 
            "timestamp": asyncio.get_event_loop().time()
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/messages/stream")
async def stream_message(
    message: UserMessage,  # Direct use of core type - no duplication!
    current_user: str = Depends(get_current_user)  # Auth from headers
):
    """
    Stream process a message and return server-sent events.
    
    AUTHENTICATION:
    - Requires valid Bearer token in Authorization header
    - User identity is extracted from token and set as message sender
    - Request body contains only message content (no auth data)
    
    FUTURE OAuth/OpenID INTEGRATION:
    - Replace get_current_user() with actual JWT verification
    - Extract user claims (email, name, roles) from verified token
    - Support rich user context and permissions
    
    REQUEST FORMAT:
    Headers:
        Authorization: Bearer <jwt_token>
        Content-Type: application/json
        
    Body (UserMessage):
        {
            "content": "Hello, how are you?",
            "role": "user"  // Optional, defaults to "user"
        }
        
    OR with rich content:
        {
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image_url": {"url": "https://..."}}
            ]
        }
    
    This endpoint:
    1. Authenticates user via headers
    2. Creates verified message with authenticated sender
    3. Queues the message for processing
    4. Returns streaming response with processing events
    5. Includes context updates, LLM responses, and tool calls
    
    Args:
        message: Message content from request body (UserMessage type)
        current_user: Authenticated user ID from Bearer token (dependency injection)
        
    Returns:
        StreamingResponse: Server-Sent Events stream of processing events
    """
    global message_queue
    
    if message_queue is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Create verified message with authenticated sender
        # Override any name in request body with verified user from token
        verified_message = UserMessage(
            content=message.content,
            name=current_user,  # Always use authenticated user identity
            # Note: role is automatically set to "user" by UserMessage class
        )
        
        # Log authenticated message for audit trail
        logger.info(f"Authenticated message from user '{current_user}' - content length: {len(str(verified_message.content))}")
        
        # Create response queue for this request (per-request isolation)
        response_queue = asyncio.Queue()
        
        # Add to processing queue with verified sender information
        await message_queue.put({
            "message": verified_message,
            "response_queue": response_queue
        })
        
        # Return streaming response
        return StreamingResponse(
            stream_response_generator(response_queue),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                # CORS headers for OAuth/frontend integration
                "Access-Control-Allow-Origin": "*",  # TODO: Configure for production
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream_message endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/configure")
async def configure_context(
    config: ContextConfig,
    current_user: str = Depends(get_current_user)  # Require auth for context changes
):
    """
    Configure the context manager with new settings.
    
    Requires authentication to prevent unauthorized context manipulation.
    Future versions may implement per-user context isolation.
    """
    global orchestrator
    
    try:
        logger.info(f"User '{current_user}' configuring context")
        
        # Create new context manager with provided config
        new_context_manager = create_context_manager(
            max_tokens=config.max_tokens,
            system_instructions=config.system_instructions,
            memory_content=config.memory_content,
            tools=config.tools
        )
        
        # Create new orchestrator with remote tool support
        new_orchestrator = await create_orchestrator(
            context_manager=new_context_manager,
            mcp_servers=config.mcp_servers,
            a2a_agents=config.a2a_agents
        )
        
        # Clean up old orchestrator and update global
        if orchestrator and hasattr(orchestrator, 'cleanup_remote_tools'):
            await orchestrator.cleanup_remote_tools()
        
        orchestrator = new_orchestrator
        
        return {
            "status": "success",
            "message": "Context configured successfully",
            "configured_by": current_user,
            "context_summary": new_context_manager.get_context_summary()
        }
        
    except Exception as e:
        logger.error(f"Error configuring context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure context: {str(e)}")


@app.get("/context/status")
async def get_context_status(
    current_user: str = Depends(get_current_user)  # Require auth for context access
):
    """Get current context status. Requires authentication."""
    global orchestrator
    
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"User '{current_user}' requesting context status")
        return orchestrator.context_manager.get_context_summary()
    except Exception as e:
        logger.error(f"Error getting context status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status(
    current_user: str = Depends(get_current_user)  # Require auth for system status
):
    """Get overall system status. Requires authentication."""
    global message_queue, orchestrator
    
    if orchestrator is None or message_queue is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        orchestrator_status_data = orchestrator.get_status()
        
        logger.info(f"User '{current_user}' requesting system status")
        
        return StatusResponse(
            orchestrator_status=orchestrator_status_data["status"],
            queue_size=message_queue.qsize(),
            context_summary=orchestrator_status_data["context_summary"],
            registered_tools=orchestrator_status_data["registered_tools"]
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/register")
async def register_tool(
    tool_name: str, 
    handler_info: Dict[str, Any],
    current_user: str = Depends(get_current_user)  # Require auth for tool registration
):
    """
    Register a tool handler (for development/testing).
    Requires authentication to prevent unauthorized tool registration.
    In production, tools would be registered during startup.
    """
    global orchestrator
    
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"User '{current_user}' registering tool: {tool_name}")
        
        # This is a simplified registration for demo purposes
        # In practice, you'd have actual handler functions
        async def mock_handler(**kwargs):
            return f"Mock result from {tool_name} with args: {kwargs}"
        
        orchestrator.register_tool_handler(tool_name, mock_handler)
        
        return {
            "status": "success",
            "message": f"Tool '{tool_name}' registered successfully",
            "registered_by": current_user,
            "registered_tools": list(orchestrator.tool_executor.get_registered_tools())
        }
    except Exception as e:
        logger.error(f"Error registering tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RemGPT API",
        "version": "0.1.0",
        "description": "Streaming conversational AI API with context management and OAuth-ready authentication",
        "authentication": {
            "type": "Bearer Token",
            "header": "Authorization: Bearer <token>",
            "future": "OAuth 2.0 / OpenID Connect support planned"
        },
        "endpoints": {
            "stream_message": "POST /messages/stream",
            "configure_context": "POST /context/configure", 
            "context_status": "GET /context/status",
            "system_status": "GET /status",
            "register_tool": "POST /tools/register"
        },
        "features": {
            "streaming": "Server-Sent Events for real-time processing",
            "authentication": "Header-based Bearer token authentication",
            "rich_content": "Support for text and image content",
            "context_management": "Dynamic context configuration and monitoring"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000) 