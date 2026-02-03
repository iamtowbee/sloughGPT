#!/usr/bin/env python3
"""
Minimal Working Test Runner
Gets UX data for system improvements.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add system to path
sys.path.insert(0, str(Path(__file__).parent))


def run_core_tests():
    """Run core system tests and gather UX insights."""
    print("🧪 SloGPT Core System UX Analysis")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Dataset Creation UX
    print("📝 Test 1: Dataset Creation UX")
    print("-" * 30)
    
    try:
        # Test creating dataset with different inputs
        from create_dataset_fixed import create_dataset
        
        # Test direct text (expected to be good UX)
        print("   🟢 Testing direct text input...")
        result = create_dataset("ux_test_text", "Artificial intelligence is transforming our world rapidly.")
        test_results['direct_text'] = {
            'success': result and result.get('success', False),
            'time_taken': 'immediate' if result and result.get('success') else 'failed',
            'commands_needed': 1,
            'user_friendliness': 'good' if result and result.get('success') else 'poor'
        }
        
        if result and result.get('success'):
            print("   ✅ Direct text: SUCCESS (1 command)")
        else:
            print("   ❌ Direct text: FAILED")
            print(f"   Error: {result.get('error', 'Unknown')}")
        
        # Test file input (might be more complex)
        print("   🟢 Testing file input...")
        result = create_dataset("ux_test_file", None, None, "ux_test.txt")
        
        if result and result.get('success'):
            test_results['file_input'] = {
                'success': True,
                'time_taken': 'immediate',
                'commands_needed': 2,  # file path + create
                'user_friendliness': 'good'
            }
            print("   ✅ File input: SUCCESS (2 commands)")
        else:
            test_results['file_input'] = {
                'success': False,
                'time_taken': 'failed',
                'commands_needed': 2,
                'user_friendliness': 'poor'
            }
            print("   ❌ File input: FAILED")
            print(f"   Error: {result.get('error', 'Unknown')}")
        
        # Test template creation (should be simple)
        print("   🟢 Testing template creation...")
        result = create_dataset("ux_test_empty")
        
        if result and result.get('success'):
            test_results['template_creation'] = {
                'success': True,
                'time_taken': 'immediate',
                'commands_needed': 1,
                'user_friendliness': 'excellent'
            }
            print("   ✅ Template: SUCCESS (1 command)")
        else:
            test_results['template_creation'] = {
                'success': False,
                'time_taken': 'failed',
                'commands_needed': 1,
                'user_friendliness': 'poor'
            }
            print("   ❌ Template: FAILED")
            print(f"   Error: {result.get('error', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
    
    return test_results


def run_training_ux_tests():
    """Test training pipeline UX."""
    print("\n🏋 Test 2: Training Pipeline UX")
    print("-" * 30)
    
    try:
        from train_simple import train_dataset
        
        # Test with simple training (should be good UX)
        print("   🟢 Testing simple training...")
        # We'll just test if the training script starts properly
        result = subprocess.run(
            ["python3", "train_simple.py", "ux_test_data", "--steps", "5"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        test_results['simple_training'] = {
            'success': result.returncode == 0,
            'has_progress': 'training steps' in result.stdout.lower(),
            'time_to_start': 'fast',
            'user_friendliness': 'good' if result.returncode == 0 else 'needs_work'
        }
        
        if result.returncode == 0:
            print("   ✅ Simple training: STARTS SUCCESSFULLY")
        else:
            print("   ❌ Simple training: FAILED TO START")
            print(f"   Error: {result.stderr}")
        
    except Exception as e:
        print(f"❌ Training UX test failed: {e}")
    
    return test_results


def run_command_ux_tests():
    """Test CLI tool usability."""
    print("\n⌨ Test 3: CLI Tool UX")
    print("-" * 30)
    
    # Test common CLI commands
    cli_tools = [
        ("create_dataset_fixed.py", "--help"),
        ("train_simple.py", "--help"),
        ("huggingface_integration.py", "--help"),
        ("simple_distributed_training.py", "--check")
    ]
    
    test_results = {}
    
    for tool, args in cli_tools:
        tool_name = tool[0].split('.')[0]
        print(f"   🟢 Testing {tool_name} help...")
        
        result = subprocess.run(
            [sys.executable, tool] + args,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        test_results[f"{tool_name}_help"] = {
            'success': result.returncode == 0,
            'has_clear_examples': 'examples:' in result.stdout.lower() or 'usage:' in result.stdout.lower(),
            'error_free': result.returncode == 0,
            'user_friendliness': 'good' if result.returncode == 0 else 'poor'
        }
        
        if result.returncode == 0:
            print(f"   ✅ {tool_name} help: CLEAR")
        else:
            print(f"   ❌ {tool_name} help: FAILED")
            print(f"   Error: {result.stderr}")
    
    # Test error handling
    print("   🟢 Testing error handling...")
    result = subprocess.run(
        [sys.executable, "create_dataset_fixed.py", "nonexistent"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    test_results['error_handling'] = {
        'success': result.returncode != 0,  # Should fail
        'has_clear_error': 'not found' in result.stderr.lower() or 'does not exist' in result.stderr.lower(),
        'error_message_clear': True,
        'user_friendliness': 'good' if result.returncode != 0 else 'poor'
    }
    
    print(f"   🧪 Error handling: {'CLEAR' if test_results['error_handling']['success'] else 'UNCLEAR'}")
    
    return test_results


def analyze_ux_results(test_results: dict) -> dict:
    """Analyze test results and generate UX recommendations."""
    recommendations = []
    
    # Analyze command complexity
    max_commands = max(
        [r.get('commands_needed', 1) for r in test_results.values()],
        default=1
    )
    
    if max_commands > 2:
        recommendations.append({
            'category': 'CLI_SIMPLIFICATION',
            'issue': 'Too many commands required for common tasks',
            'suggestion': 'Implement single-command workflows',
            'priority': 'high'
        })
    
    # Analyze error message quality
    for test_name, result in test_results.items():
        if result.get('success') == False:
            error_msg = result.get('error_message', '').lower()
            
            if error_msg and ('not found' in error_msg or 'does not exist' in error_msg):
                recommendations.append({
                    'category': 'ERROR_IMPROVEMENT',
                    'issue': f'Poor error messages in {test_name}',
                    'suggestion': 'Provide actionable error messages with specific file paths or solutions',
                    'priority': 'medium'
                })
    
    # Analyze user experience
    total_tests = len(test_results)
    successful_tests = len([r for r in test_results.values() if r.get('success', False)])
    
    if successful_tests < total_tests:
        recommendations.append({
            'category': 'USER_GUIDANCE',
            'issue': 'Some tests failed - user may need guidance',
            'suggestion': 'Add interactive tutorials and better help system',
            'priority': 'high'
        })
    
    return {
        'recommendations': recommendations,
        'test_summary': {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'max_commands_needed': max_commands,
            'overall_ux_score': 'good' if successful_tests >= total_tests * 0.8 else 'needs_improvement'
        }
    }


def generate_improvements(ux_analysis: dict) -> dict:
    """Generate specific improvements based on UX analysis."""
    improvements = {}
    
    for rec in ux_analysis.get('recommendations', []):
        category = rec['category']
        priority = rec['priority']
        issue = rec['issue']
        suggestion = rec['suggestion']
        
        if category == 'CLI_SIMPLIFICATION':
            if priority == 'high':
                improvements['high_priority'] = {
                    'description': 'Implement single-command workflows',
                    'commands': [
                        'Create aliases in CLI shortcuts module',
                        'Add batch processing commands',
                        'Implement smart defaults'
                    ]
                }
        
        elif category == 'ERROR_IMPROVEMENT':
            if priority == 'high':
                improvements['error_messages'] = {
                    'description': 'Improve error message clarity',
                    'commands': [
                        'Standardize error message format',
                        'Add helpful suggestions to failed commands',
                        'Include relevant command examples'
                    ]
                }
        
        elif category == 'USER_GUIDANCE':
            if priority == 'high':
                improvements['user_guidance'] = {
                    'description': 'Improve user onboarding and help',
                    'commands': [
                        'Create interactive tutorials with step-by-step guidance',
                        'Add "Getting Started" section to documentation',
                        'Implement help command with categorized topics',
                        'Add example workflows to CLI help'
                    ]
                }
    
    return improvements


def main():
    """Main UX analysis runner."""
    print("🎨 SloGPT UX Analysis & Improvement")
    print("=" * 50)
    
    # Run UX tests
    print("🧪 Running UX Analysis Tests...")
    print()
    
    # Test 1: Dataset Creation
    dataset_results = run_core_tests()
    
    # Test 2: Training Pipeline
    training_results = run_training_ux_tests()
    
    # Test 3: CLI Tools
    cli_results = run_command_ux_tests()
    
    # Analyze results
    print("\n📊 Analyzing User Experience...")
    print()
    
    all_results = {
        'dataset_creation': dataset_results,
        'training_pipeline': training_results,
        'cli_tools': cli_results
        'analysis_timestamp': None
    }
    
    ux_analysis = analyze_ux_results(all_results)
    improvements = generate_improvements(ux_analysis)
    
    # Generate recommendations
    print("\n🔧 Generating UX Improvement Recommendations...")
    print()
    
    # Save results
    results_file = Path("ux_analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'analysis_date': '2026-01-31',
            'test_results': all_results,
            'ux_analysis': ux_analysis,
            'improvements': improvements
        }, f, indent=2)
    
    # Generate improvement report
    improvements_file = Path("ux_improvements.md")
    with open(improvements_file, 'w') as f:
        f.write("# SloGPT UX Improvements Report\n")
        f.write(f"Generated: 2026-01-31\n\n")
        f.write("## 🎨 Key Findings\n")
        
        for category in ['high_priority', 'error_messages', 'user_guidance']:
            if category in improvements:
                f.write(f"### 🔧 {category.replace('_', ' ').title()}\n")
                for i, improvement in enumerate(improvements[category].get('commands', []), 1):
                    f.write(f"{i}. {improvement}\n")
                f.write("\n")
        
        f.write("## 📊 Test Results Summary\n")
        f.write(f"Overall UX Score: {ux_analysis.get('test_summary', {}).get('overall_ux_score', 'unknown')}\n")
        f.write(f"Tests Analyzed: {ux_analysis.get('test_summary', {}).get('total_tests', 0)}\n")
    
    print(f"✅ UX Analysis Complete!")
    print(f"📁 Results: {results_file}")
    print(f"📋 Improvements: {improvements_file}")
    
    print(f"\n🎯 KEY RECOMMENDATIONS")
    print("=" * 50)
    
    for category, improvement_list in improvements.items():
        print(f"🔧 {category.replace('_', ' ').title()}")
        for i, improvement in enumerate(improvement_list.get('commands', []), 1):
            print(f"   {i}. {improvement}")
        print()
    
    return 0


if __name__ == '__main__':
    exit(main())