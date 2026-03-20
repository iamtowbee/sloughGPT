#!/usr/bin/env python3
"""
SloughGPT SDK CLI
Command-line interface for the SloughGPT SDK.

Usage:
    sloughgpt-cli --help
    sloughgpt-cli health
    sloughgpt-cli generate "Hello, how are you?"
    sloughgpt-cli chat "What is Python?"
    sloughgpt-cli models
"""

import argparse
import json
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sloughgpt_sdk import SloughGPTClient, ChatMessage, APIKeyManager, KeyTier
except ImportError:
    print("Error: sloughgpt-sdk not found. Install with: pip install sloughgpt-sdk")
    sys.exit(1)


def format_json(data) -> str:
    """Format data as colored JSON."""
    return json.dumps(data, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="SloughGPT SDK CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--url", "-u",
        default=os.environ.get("SLOUGHGPT_API_URL", "http://localhost:8000"),
        help="API base URL"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.environ.get("SLOUGHGPT_API_KEY"),
        help="API key for authentication"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt2",
        help="Default model to use"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", "-T",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    health = subparsers.add_parser("health", help="Check API health")
    info = subparsers.add_parser("info", help="Get system info")
    
    gen = subparsers.add_parser("generate", help="Generate text")
    gen.add_argument("prompt", help="Prompt for generation")
    gen.add_argument("--stream", "-s", action="store_true", help="Stream output")
    
    chat = subparsers.add_parser("chat", help="Chat completion")
    chat.add_argument("message", help="User message")
    chat.add_argument("--system", help="System prompt")
    
    models = subparsers.add_parser("models", help="List available models")
    models.add_argument("--hf", action="store_true", help="Include HuggingFace models")
    
    datasets = subparsers.add_parser("datasets", help="List available datasets")
    
    metrics = subparsers.add_parser("metrics", help="Get API metrics")
    metrics.add_argument("--prometheus", action="store_true", help="Prometheus format")
    
    batch = subparsers.add_parser("batch", help="Batch generation")
    batch.add_argument("prompts", nargs="+", help="List of prompts")
    
    key = subparsers.add_parser("key", help="API key management")
    key_subparsers = key.add_subparsers(dest="key_action", help="Key actions")
    
    key_create = key_subparsers.add_parser("create", help="Create a new API key")
    key_create.add_argument("--name", "-n", required=True, help="Key name")
    key_create.add_argument("--tier", "-t", default="free", choices=["free", "starter", "pro", "enterprise"], help="Subscription tier")
    key_create.add_argument("--expires", "-e", type=int, help="Days until expiration")
    key_create.add_argument("--quota-daily", type=int, help="Daily quota limit")
    key_create.add_argument("--quota-monthly", type=int, help="Monthly quota limit")
    
    key_list = key_subparsers.add_parser("list", help="List API keys")
    
    key_info = key_subparsers.add_parser("info", help="Get key info")
    key_info.add_argument("key_id", help="Key ID")
    
    key_rotate = key_subparsers.add_parser("rotate", help="Rotate API key")
    key_rotate.add_argument("key_id", help="Key ID to rotate")
    
    key_revoke = key_subparsers.add_parser("revoke", help="Revoke API key")
    key_revoke.add_argument("key_id", help="Key ID to revoke")
    
    key_delete = key_subparsers.add_parser("delete", help="Delete API key")
    key_delete.add_argument("key_id", help="Key ID to delete")
    
    key_usage = key_subparsers.add_parser("usage", help="Get key usage stats")
    key_usage.add_argument("key_id", help="Key ID")
    
    key_reset = key_subparsers.add_parser("reset", help="Reset key usage")
    key_reset.add_argument("key_id", help="Key ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        client = SloughGPTClient(
            base_url=args.url,
            api_key=args.api_key,
            timeout=60,
        )
        
        if args.command == "health":
            result = client.health()
            if args.json:
                print(format_json(result.raw))
            else:
                print(f"Status: {result.status}")
                print(f"Version: {result.version}")
                print(f"Model Loaded: {result.model_loaded}")
                if result.model_name:
                    print(f"Model: {result.model_name}")
                print(f"Device: {result.device}")
        
        elif args.command == "info":
            result = client.info()
            if args.json:
                print(format_json(result.raw))
            else:
                print(f"Version: {result.version}")
                print(f"PyTorch: {result.pytorch_version}")
                print(f"CUDA Available: {result.cuda_available}")
                if result.cuda:
                    print(f"GPU: {result.cuda.get('device', 'N/A')}")
                print(f"CPU Cores: {result.cpu_count}")
        
        elif args.command == "generate":
            if args.stream:
                print("Generating (stream): ", end="", flush=True)
                for token in client.generate_stream(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                ):
                    print(token, end="", flush=True)
                print()
            else:
                result = client.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                if args.json:
                    print(format_json(result.raw_response))
                else:
                    print(result.generated_text)
                
                if args.verbose:
                    print(f"\nTokens: {result.tokens_generated}")
                    print(f"Time: {result.inference_time_ms:.2f}ms")
        
        elif args.command == "chat":
            messages = []
            if args.system:
                messages.append(ChatMessage.system(args.system))
            messages.append(ChatMessage.user(args.message))
            
            result = client.chat(
                messages,
                model=args.model,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            
            if args.json:
                print(format_json(result.raw_response))
            else:
                print(result.message.content)
        
        elif args.command == "models":
            models_list = client.list_models()
            if args.json:
                print(format_json([m.raw for m in models_list]))
            else:
                for m in models_list:
                    print(f"  {m.id}")
                    if m.source:
                        print(f"    Source: {m.source}")
                    if m.description:
                        print(f"    {m.description}")
        
        elif args.command == "datasets":
            datasets_list = client.list_datasets()
            if args.json:
                print(format_json([d.raw for d in datasets_list]))
            else:
                for d in datasets_list:
                    print(f"  {d.id}")
                    if d.description:
                        print(f"    {d.description}")
        
        elif args.command == "metrics":
            if args.prometheus:
                print(client.metrics_prometheus())
            else:
                result = client.metrics()
                if args.json:
                    print(format_json(result.raw))
                else:
                    print(f"Total Requests: {result.requests_total}")
                    print(f"Successful: {result.requests_success}")
                    print(f"Failed: {result.requests_failed}")
                    print(f"Cache Hits: {result.cache_hits}")
                    print(f"Cache Misses: {result.cache_misses}")
                    print(f"Avg Response Time: {result.avg_response_time_ms:.2f}ms")
        
        elif args.command == "batch":
            result = client.batch_generate(
                args.prompts,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            if args.json:
                print(format_json({
                    "total": result.total_prompts,
                    "successful": result.successful,
                    "failed": result.failed,
                    "results": [r.raw_response for r in result.results]
                }))
            else:
                print(f"Processed {result.total_prompts} prompts")
                print(f"Successful: {result.successful}")
                print(f"Failed: {result.failed}")
                for i, r in enumerate(result.results):
                    print(f"\n{i+1}. {args.prompts[i][:50]}...")
                    print(f"   {r.generated_text[:100]}...")
        
        elif args.command == "key":
            key_manager = APIKeyManager()
            
            if args.key_action == "create":
                tier = KeyTier(args.tier)
                new_key, key_data = key_manager.create_key(
                    name=args.name,
                    tier=tier,
                    expires_in_days=args.expires,
                    quota_daily=args.quota_daily,
                    quota_monthly=args.quota_monthly,
                )
                print("\n" + "="*50)
                print("NEW API KEY CREATED")
                print("="*50)
                print(f"\nKey: {new_key}")
                print(f"Key ID: {key_data.key_id}")
                print(f"Name: {key_data.name}")
                print(f"Tier: {key_data.tier.value}")
                print(f"Rate Limit: {key_data.rate_limit}/min")
                print(f"Daily Quota: {key_data.quota_daily}")
                print(f"Monthly Quota: {key_data.quota_monthly}")
                print("\n⚠️  Save this key securely - it will not be shown again!")
                print("="*50)
            
            elif args.key_action == "list":
                keys = key_manager.list_keys()
                if not keys:
                    print("No API keys found.")
                else:
                    for k in keys:
                        status = "✓" if k.is_valid() else "✗"
                        print(f"\n{status} {k.name} ({k.key_id})")
                        print(f"   Prefix: {k.prefix}...")
                        print(f"   Tier: {k.tier.value}")
                        print(f"   Active: {k.is_active}")
                        print(f"   Usage: {k.usage_today}/{k.quota_daily} today, {k.usage_this_month}/{k.quota_monthly} this month")
            
            elif args.key_action == "info":
                key_data = key_manager.get_key_info(args.key_id)
                if key_data:
                    print(format_json(key_data.to_dict()))
                else:
                    print(f"Key not found: {args.key_id}")
            
            elif args.key_action == "rotate":
                try:
                    new_key, new_data = key_manager.rotate_key(args.key_id)
                    print("\n" + "="*50)
                    print("KEY ROTATED")
                    print("="*50)
                    print(f"\nNew Key: {new_key}")
                    print(f"Key ID: {new_data.key_id}")
                    print(f"Name: {new_data.name}")
                    print("\n⚠️  The old key has been revoked!")
                    print("⚠️  Save this new key securely!")
                    print("="*50)
                except ValueError as e:
                    print(f"Error: {e}")
            
            elif args.key_action == "revoke":
                if key_manager.revoke_key(args.key_id):
                    print(f"Key revoked: {args.key_id}")
                else:
                    print(f"Key not found: {args.key_id}")
            
            elif args.key_action == "delete":
                if key_manager.delete_key(args.key_id):
                    print(f"Key deleted: {args.key_id}")
                else:
                    print(f"Key not found: {args.key_id}")
            
            elif args.key_action == "usage":
                stats = key_manager.get_usage_stats(args.key_id)
                if stats:
                    print(f"\nUsage Statistics for {args.key_id}:")
                    print(f"  Total Requests: {stats['total_requests']}")
                    print(f"  Today: {stats['requests_today']}/{stats['daily_limit']} ({stats['daily_usage_percent']:.1f}%)")
                    print(f"  This Month: {stats['requests_this_month']}/{stats['monthly_limit']} ({stats['monthly_usage_percent']:.1f}%)")
                    print(f"  Remaining Today: {stats['daily_remaining']}")
                    print(f"  Remaining This Month: {stats['monthly_remaining']}")
                else:
                    print(f"Key not found: {args.key_id}")
            
            elif args.key_action == "reset":
                if key_manager.reset_usage(args.key_id):
                    print(f"Usage reset for: {args.key_id}")
                else:
                    print(f"Key not found: {args.key_id}")
            
            else:
                key.print_help()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
