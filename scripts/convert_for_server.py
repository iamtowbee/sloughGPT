#!/usr/bin/env python3
"""
Create a server-compatible checkpoint from sloughgpt_finetuned.pt
"""

import sys
import os
import torch
from pathlib import Path

# Add sloughgpt to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def convert_for_server():
    model_path = Path(__file__).parent.parent / "models" / "sloughgpt_finetuned.pt"
    output_path = Path(__file__).parent.parent / "models" / "sloughgpt_server.pt"
    
    print(f"Loading model from {model_path}...")
    
    # Load the safetensors checkpoint
    try:
        # Try loading as safetensors (zip format)
        import zipfile
        import pickle
        
        with zipfile.ZipFile(model_path, 'r') as z:
            # Read the pickle data
            with z.open(f"{model_path.stem}/data.pkl") as f:
                data = pickle.load(f)
        
        print(f"Loaded data type: {type(data)}")
        print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        
        # Create server-compatible format
        checkpoint = {
            "model": data.get("model_state_dict", data),
            "stoi": data.get("stoi", {chr(i): i for i in range(256)}),
            "itos": data.get("itos", {i: chr(i) for i in range(256)}),
            "config": data.get("training_info", {}),
            "chars": list(range(256)),
        }
        
        # If model is nested under a key
        if "model_state" in data:
            checkpoint["model"] = data["model_state"]
        
        print(f"Saving server-compatible checkpoint to {output_path}...")
        torch.save(checkpoint, output_path)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: load the checkpoint differently
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Reformat for server
            server_ckpt = {
                "model": checkpoint,
                "stoi": {chr(i): i for i in range(256)},
                "itos": {i: chr(i) for i in range(256)},
                "chars": list(range(256)),
            }
            
            torch.save(server_ckpt, output_path)
            print(f"Saved to {output_path}")
        except Exception as e2:
            print(f"Alternative approach failed: {e2}")

if __name__ == "__main__":
    convert_for_server()
