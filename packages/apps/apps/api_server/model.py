"""Simple mock model for API server testing"""

class MockGPT:
    """Mock GPT model for testing API functionality"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/slogpt.pt"
        self.loaded = False
        
    async def load_model(self):
        """Mock model loading"""
        self.loaded = True
        
    async def generate(self, prompt: str, **kwargs):
        """Mock text generation"""
        return f"Mock response to: {prompt[:50]}..."
        
    async def unload_model(self):
        """Mock model unloading"""
        self.loaded = False

# Create instance
GPT = MockGPT