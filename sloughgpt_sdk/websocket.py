"""
SloughGPT SDK - WebSocket Client
WebSocket client for real-time streaming.
"""

import json
import threading
from typing import Optional, Callable, Dict, Any, List, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    pass


@dataclass
class WebSocketMessage:
    """A message received via WebSocket."""
    type: str
    data: Any
    raw: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSocketMessage":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "unknown"),
            data=data.get("data", ""),
            raw=data,
        )


class WebSocketClient:
    """
    WebSocket client for SloughGPT real-time streaming.
    
    Example:
    
    ```python
    from sloughgpt_sdk.websocket import WebSocketClient
    
    def on_message(msg):
        print(f"Received: {msg.data}")
    
    def on_complete():
        print("Generation complete!")
    
    ws = WebSocketClient("http://localhost:8000")
    ws.connect()
    ws.send_generate("Hello, how are you?", on_message=on_message, on_complete=on_complete)
    ws.close()
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        """Initialize WebSocket client."""
        self.base_url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
        self.api_key = api_key
        self._ws = None
        self._connected = False
        self._handlers: Dict[str, List[Callable]] = {}
    
    def connect(self) -> bool:
        """Connect to WebSocket endpoint."""
        try:
            import websocket
            
            headers = []
            if self.api_key:
                headers.append(f"X-API-Key: {self.api_key}")
            
            self._ws = websocket.WebSocketApp(
                f"{self.base_url}/ws",
                header=headers,
                on_message=self._handle_message,
                on_error=self._handle_error,
                on_open=self._handle_open,
                on_close=self._handle_close,
            )
            
            thread = threading.Thread(target=self._run_forever, daemon=True)
            thread.start()
            
            return True
        except ImportError:
            raise ImportError("websocket-client package required: pip install websocket-client")
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")
    
    def _run_forever(self):
        """Run WebSocket in background thread."""
        if self._ws:
            self._ws.run_forever(ping_timeout=30)
    
    def _handle_message(self, ws, message):
        """Handle incoming message."""
        try:
            data = json.loads(message)
            msg = WebSocketMessage.from_dict(data)
            
            if msg.type in self._handlers:
                for handler in self._handlers[msg.type]:
                    handler(msg)
            if "*" in self._handlers:
                for handler in self._handlers["*"]:
                    handler(msg)
        except json.JSONDecodeError:
            pass
    
    def _handle_error(self, ws, error):
        """Handle WebSocket error."""
        pass
    
    def _handle_open(self, ws):
        """Handle connection open."""
        self._connected = True
    
    def _handle_close(self, ws, close_status_code, close_msg):
        """Handle connection close."""
        self._connected = False
    
    def on(self, event_type: str, handler: Callable[[WebSocketMessage], None]):
        """Register event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def send_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        **kwargs
    ):
        """Send generation request."""
        if not self._connected:
            raise ConnectionError("Not connected. Call connect() first.")
        
        message = {
            "type": "generate",
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs
        }
        self._ws.send(json.dumps(message))
    
    def send_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Send chat request."""
        if not self._connected:
            raise ConnectionError("Not connected. Call connect() first.")
        
        message = {
            "type": "chat",
            "messages": messages,
            **kwargs
        }
        self._ws.send(json.dumps(message))
    
    def send_ping(self):
        """Send ping to keep connection alive."""
        if self._connected:
            self._ws.send(json.dumps({"type": "ping"}))
    
    def close(self):
        """Close WebSocket connection."""
        if self._ws:
            self._ws.close()
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


class StreamIterator:
    """Iterator for streaming tokens."""
    
    def __init__(self, generator: "StreamGenerator"):
        """Initialize iterator."""
        self.generator = generator
        self._index = 0
    
    def __iter__(self):
        """Return iterator."""
        return self
    
    def __next__(self) -> str:
        """Get next token."""
        import time
        while self._index >= len(self.generator._buffer) and not self.generator._complete:
            time.sleep(0.01)
        
        if self._index >= len(self.generator._buffer):
            raise StopIteration
        
        token = self.generator._buffer[self._index]
        self._index += 1
        return token
    
    def __aiter__(self):
        """Return async iterator."""
        return self
    
    async def __anext__(self):
        """Get next token asynchronously."""
        import asyncio
        while self._index >= len(self.generator._buffer) and not self.generator._complete:
            await asyncio.sleep(0.01)
        
        if self._index >= len(self.generator._buffer):
            raise StopAsyncIteration
        
        token = self.generator._buffer[self._index]
        self._index += 1
        return token


class StreamGenerator:
    """
    Context manager for streaming responses.
    
    Example:
    
    ```python
    from sloughgpt_sdk.websocket import StreamGenerator
    
    with StreamGenerator("http://localhost:8000") as stream:
        for token in stream.generate("Hello"):
            print(token, end="", flush=True)
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        """Initialize stream generator."""
        self.client = WebSocketClient(base_url, api_key)
        self._buffer = []
        self._complete = False
    
    def __enter__(self):
        """Enter context manager."""
        self.client.connect()
        
        def on_token(msg):
            if msg.type == "token":
                self._buffer.append(msg.data)
            elif msg.type == "complete":
                self._complete = True
        
        self.client.on("token", on_token)
        self.client.on("complete", on_token)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.client.close()
    
    def generate(self, prompt: str, **kwargs) -> StreamIterator:
        """Generate with streaming."""
        self._buffer = []
        self._complete = False
        self.client.send_generate(prompt, **kwargs)
        return StreamIterator(self)
    
    def get_full_text(self) -> str:
        """Get full generated text."""
        return "".join(self._buffer)
