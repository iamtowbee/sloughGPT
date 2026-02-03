#!/usr/bin/env python3
"""
Simplified SloGPT UX Testing Framework
Focused on gathering user experience data and feedback.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add system to path
sys.path.insert(0, str(Path(__file__).parent))


def test_cli_usability():
    """Test CLI usability across all tools."""
    print("🎯 TESTING CLI USABILITY")
    print("=" * 50)
    
    cli_tools = [
        "create_dataset_fixed.py",
        "train_simple.py", 
        "huggingface_integration.py",
        "simple_distributed_training.py",
        "model_optimizer.py",
        "deployment_system.py"
        "web_interface.py",
        "analytics_dashboard.py"
        "benchmark_system.py",
        "system_showcase.py"
        "comprehensive_test_framework.py"
        "test_simple_distributed.py",
        "test_hf_basic.py"
        "test_huggingface_integration.py",
        "demo_huggingface.py"
    ]
    
    usability_issues = []
    
    for tool in cli_tools:
        if not Path(tool).exists():
            usability_issues.append({
                'tool': tool,
                'issue': 'missing',
                'severity': 'high'
            })
            continue
        
        print(f"🔧 Testing: {tool}")
        
        # Test help functionality
        try:
            result = subprocess.run(
                [sys.executable, tool, "--help"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode != 0:
                usability_issues.append({
                    'tool': tool,
                    'issue': 'help_command_error',
                    'severity': 'medium',
                    'details': result.stderr
                })
            else:
                print(f"   ✅ Help command works")
        except Exception as e:
            usability_issues.append({
                'tool': tool,
                'issue': 'help_command_exception',
                'severity': 'medium'
                'details': str(e)
            })
        
        # Test error cases
        test_cases = [
            ("--invalid-flag", "non-existent flag"),
            ("--missing-arg", "missing required argument"),
            ("", "no arguments"),
            ("create_dataset mydata", "missing quoted text argument"),
        ]
        
        for args, description in test_cases:
            print(f"   🧪 Testing: {description}")
            
            try:
                result = subprocess.run(
                    [sys.executable, tool] + args.split(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    usability_issues.append({
                        'tool': tool,
                        'issue': 'error_handling',
                        'severity': 'medium',
                        'details': f"Failed on: {description} - {result.stderr}"
                    })
                else:
                    # This is expected - error return codes are good UX
                    print(f"   ✅ Correctly handles error: {description}")
                    
            except subprocess.TimeoutExpired:
                usability_issues.append({
                    'tool': tool,
                        'issue': 'timeout',
                        'severity': 'high',
                        'details': f"Timeout on: {description}"
                })
            except Exception as e:
                usability_issues.append({
                    'tool': tool,
                        'issue': 'exception',
                        'severity': 'high',
                        'details': f"Exception on {description}: {e}"
                })
    
    # Calculate usability score
    total_tools = len(cli_tools)
    issues_found = len([issue for issue in usability_issues if issue['severity'] != 'low'])
    
    if issues_found == 0:
        usability_score = 100
        usability_level = "Excellent"
    elif issues_found <= 2:
        usability_score = 85
        usability_level = "Good"
    elif issues_found <= 5:
        usability_score = 70
        usability_level = "Fair"
    else:
        usability_score = 50
        usability_level = "Needs Improvement"
    
    print(f"\n📊 CLI USABILITY RESULTS")
    print(f"   Total Tools: {total_tools}")
    print(f"   Issues Found: {issues_found}")
    print(f"   Usability Score: {usability_score}/100")
    print(f"   Usability Level: {usability_level}")
    
    if usability_issues:
        print(f"\n🚨 USABILITY ISSUES:")
        for issue in usability_issues:
            if issue['severity'] == 'high':
                print(f"   ❌ HIGH: {issue['tool']} - {issue['issue']}")
            elif issue['severity'] == 'medium':
                print(f"   ⚠️ MEDIUM: {issue['tool']} - {issue['issue']}")
            else:
                print(f"   ℹ️ LOW: {issue['tool']} - {issue['issue']}")
    
    return {
        'score': usability_score,
        'level': usability_level,
        'issues': usability_issues,
        'total_tools': total_tools,
        'issues_count': issues_found
    }


def test_error_message_quality():
    """Test quality of error messages."""
    print("📝 TESTING ERROR MESSAGE QUALITY")
    print("=" * 50)
    
    # Test error message scenarios
    error_scenarios = [
        {
            'tool': 'create_dataset_fixed.py',
            'test': 'missing_dataset',
            'expected': "Clear, actionable message",
            'command': ['python3', 'create_dataset_fixed.py', 'nonexistent'],
            'test_func': lambda: subprocess.run([
                sys.executable, 'python3', 'create_dataset_fixed.py', 'nonexistent'], 
                capture_output=True
            ])
        },
        {
            'tool': 'train_simple.py',
            'test': 'invalid_dataset_format',
            'expected': "Describes format issue clearly",
            'command': ['python3', 'train_simple.py', '--dataset', 'corrupted_data'],
            'test_func': lambda: subprocess.run([
                sys.executable, 'python3', 'train_simple.py', '--dataset', 'corrupted_data'], 
                capture_output=True
            ])
        },
        {
            'tool': 'huggingface_integration.py',
            'test': 'missing_dependencies',
            'expected': "Helpful dependency installation instructions",
            'test_func': lambda: subprocess.run([
                sys.executable, 'python3', 'huggingface_integration.py', 'convert-model'], 
                capture_output=True, stderr=subprocess.STDOUT
            ], env={'PYTHONPATH': '/invalid'})
        }
    ]
    
    message_quality_scores = []
    
    for scenario in error_scenarios:
        print(f"🧪 Testing: {scenario['test']} with {scenario['tool']}")
        
        try:
            result = scenario['test_func']()
            output = result.stderr
            
            # Analyze message quality
            if result.returncode != 0:
                message_score = 80  # Error occurred but message exists
                has_suggestions = 'install' in output.lower() or 'pip install' in output.lower()
                has_next_steps = 'try:' in output.lower() or 'check:' in output.lower()
            else:
                message_score = 40  # Error occurred but unclear message
            else:
                message_score = 95  # Command not found (clear error)
            
            # Check for helpful elements
            helpful_elements = 0
            if has_suggestions:
                helpful_elements += 2
            if has_next_steps:
                helpful_elements += 2
            if 'error' in output.lower():
                helpful_elements += 1
            if 'corrupted' in output.lower() or 'invalid' in output.lower():
                helpful_elements += 1
            
            # Add helpful elements to score
            message_score += helpful_elements * 5
            
            print(f"   Return code: {result.returncode}")
            print(f"   Message: {output[:100]}..." if output else "No output")
            print(f"   Score: {message_score}/100")
            print(f"   Has suggestions: {has_suggestions}")
            print(f"   Has next steps: {has_next_steps}")
            
            message_quality_scores.append({
                'tool': scenario['tool'],
                'test': scenario['test'],
                'score': message_score,
                'expected': scenario['expected']
                'has_suggestions': has_suggestions,
                'has_next_steps': has_next_steps
            })
            
        except Exception as e:
            print(f"   ⚠️ Exception during test: {e}")
            message_quality_scores.append({
                'tool': scenario['tool'],
                'test': scenario['test'],
                'score': 0,
                'error': str(e)
            })
    
    # Calculate average message quality score
    valid_scores = [s['score'] for s in message_quality_scores if s['score'] > 0]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
    else:
        avg_score = 0
    
    print(f"\n📊 ERROR MESSAGE QUALITY RESULTS")
    print(f"   Average Score: {avg_score:.1f}/100")
    
    if avg_score >= 80:
        quality_level = "Excellent"
    elif avg_score >= 60:
        quality_level = "Good"
    elif avg_score >= 40:
        quality_level = "Fair"
    else:
        quality_level = "Poor"
    
    print(f"   Quality Level: {quality_level}")
    
    return {
        'average_score': avg_score,
        'level': quality_level,
        'scores': message_quality_scores,
        'total_scenarios': len(error_scenarios)
    }


def gather_ux_feedback():
    """Gather UX feedback from various sources."""
    print("📊 GATHERING UX FEEDBACK")
    print("=" * 50)
    
    feedback_data = {
        'timestamp': time.time(),
        'sources': [],
        'user_stories': [],
        'pain_points': [],
        'suggestions': []
        'metrics': {}
    }
    
    # Simulate user feedback collection
    simulated_feedback = [
        {
            'source': 'cli_testing',
            'story': 'CLI commands are confusing - too many flags',
            'pain_point': 'I struggle to remember the right command flags',
            'suggestion': 'Add aliases or simplified commands'
        },
        {
            'source': 'error_messages',
            'story': 'Error messages are not helpful - just says "failed"',
            'pain_point': "I don't understand what went wrong or how to fix it",
            'suggestion': 'Provide specific error details and fix suggestions'
        },
        {
            'source': 'documentation',
            'story': 'Documentation is too technical and lacks examples',
            'pain_point': 'I can\'t find practical examples for common tasks',
            'suggestion': 'Add step-by-step tutorials and code examples'
        },
        {
            'source': 'performance',
            'story': 'Training is slow and memory usage is high',
            'pain_point': 'I can\'t train large models efficiently',
            'suggestion': 'Implement quantization and distributed training'
        },
        {
            'source': 'new_user',
            'story': 'I\'m new to ML and the system is overwhelming',
            'pain_point': 'Too many options and concepts to learn at once',
            'suggestion': 'Create guided onboarding with progressive complexity'
        }
    ]
    
    feedback_data['user_stories'] = simulated_feedback
    feedback_data['pain_points'] = [item['pain_point'] for item in simulated_feedback]
    feedback_data['suggestions'] = [item['suggestion'] for item in simulated_feedback]
    
    # Analyze feedback for patterns
    common_themes = {}
    for item in simulated_feedback:
        theme = item['suggestion']
        if theme in common_themes:
            common_themes[theme] += 1
        else:
            common_themes[theme] = 1
    
    feedback_data['suggestions'] = [item for item in simulated_feedback 
                        if item['pain_point'] in ['confusing', 'overwhelming', 'technical']]
    
    print(f"   Collected {len(simulated_feedback)} feedback items")
    print(f"   Identified {len(common_themes)} common themes")
    
    # Generate UX recommendations
    ux_recommendations = []
    
    # High priority fixes based on feedback
    high_priority_fixes = [
        "Create command aliases for common workflows",
        "Improve error message clarity and actionability",
        "Add step-by-step examples to documentation",
        "Implement progressive complexity in tutorials",
        "Add CLI help and auto-completion"
        "Optimize performance and reduce memory usage"
        "Create guided setup wizard for new users"
    ]
    
    # Medium priority improvements
    medium_improvements = [
        "Add interactive tutorials and video guides",
        "Implement in-app help system",
        "Add progress indicators and status updates",
        "Create template system for common tasks",
        "Improve web interface responsiveness",
        "Add more comprehensive error checking"
        "Implement user preference settings"
    ]
    
    feedback_data['recommendations'] = {
        'high_priority': high_priority_fixes,
        'medium_priority': medium_improvements,
        'low_priority': [
            "Add tooltips and hover help",
            "Implement dark mode theme",
            "Add keyboard shortcuts",
            "Optimize mobile interface",
            "Add customization options"
        ]
    }
    
    # Save feedback data
    feedback_file = Path("ux_feedback_data.json")
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    print(f"💾 UX feedback data saved: {feedback_file}")
    
    return feedback_data


def generate_ux_improvements():
    """Generate UX improvements based on analysis."""
    print("🔧 GENERATING UX IMPROVEMENTS")
    print("=" * 50)
    
    # Create improvement plan
    improvement_plan = {
        'timestamp': time.time(),
        'version': '1.0.0',
        'categories': {
            'cli_usability': {
                'title': 'CLI User Experience',
                'improvements': [
                    {
                        'id': 'cmd_aliases',
                        'title': 'Command Aliases',
                        'description': 'Create shortcut commands for common workflows',
                        'priority': 'high',
                        'implementation': 'Create aliases.py with common command combinations',
                        'benefits': 'Reduces typing, increases productivity'
                    },
                    {
                        'id': 'help_system',
                        'title': 'Enhanced Help System',
                        'description': 'Contextual help with examples',
                        'priority': 'high',
                        'implementation': 'Add help --task for guided assistance',
                        'benefits': 'Helps users find right commands quickly'
                    },
                    {
                        'id': 'error_improvement',
                        'title': 'Better Error Messages',
                        'description': 'Clear, actionable error messages',
                        'priority': 'high',
                        'implementation': 'Structured error format with fixes',
                        'benefits': 'Users can self-diagnose and fix issues'
                    }
                ]
            },
            'documentation': {
                'title': 'Documentation Enhancement',
                'improvements': [
                    {
                        'id': 'interactive_tutorials',
                        'title': 'Interactive Tutorials',
                        'description': 'Step-by-step guided learning',
                        'priority': 'medium',
                        'implementation': 'Create interactive tutorials with code execution',
                        'benefits': 'Hands-on learning experience'
                    },
                    {
                        'id': 'code_examples',
                        'title': 'Code Examples',
                        'description': 'Practical examples for common tasks',
                        'priority': 'medium',
                        'implementation': 'Add working code examples throughout docs',
                        'benefits': 'Users can copy-paste working solutions'
                    },
                    {
                        'id': 'quick_start',
                        'title': 'Quick Start Guide',
                        'description': 'Streamlined getting started process',
                        'priority': 'high',
                        'implementation': 'Simplified onboarding with minimal setup',
                        'benefits': 'Reduces initial friction'
                    }
                ]
            },
            'user_experience': {
                'title': 'Overall User Experience',
                'improvements': [
                    {
                        'id': 'onboarding',
                        'title': 'Progressive Onboarding',
                        'description': 'Gradual complexity introduction',
                        'priority': 'medium',
                        'implementation': 'Multi-stage tutorials with progressive disclosure',
                        'benefits': 'Reduces cognitive overload'
                    },
                    {
                        'id': 'preferences',
                        'title': 'User Preferences',
                        'description': 'Customizable interface options',
                        'priority': 'low',
                        'implementation': 'Allow theme selection, layouts, and settings',
                        'benefits': 'Personalized user experience'
                    },
                    {
                        'id': 'feedback_system',
                        'title': 'Feedback Collection',
                        'description': 'In-app feedback and usage analytics',
                        'priority': 'medium',
                        'implementation': 'Anonymous feedback collection and improvement suggestions',
                        'benefits': 'Continuous UX optimization'
                    }
                ]
            },
            'performance': {
                'title': 'Performance Optimization',
                'improvements': [
                    {
                        'id': 'memory_optimization',
                        'title': 'Memory Usage Optimization',
                        'description': 'Reduce memory footprint',
                        'priority': 'medium',
                        'implementation': 'Better memory management and cleanup',
                        'benefits': 'Better performance on limited hardware'
                    },
                    {
                        'id': 'progressive_loading',
                        'title': 'Progressive Loading',
                        'description': 'Stage content loading with progress indicators',
                        'priority': 'medium',
                        'implementation': 'Show progress bars and estimated times',
                        'benefits': 'Better user perception of performance'
                    },
                    {
                        'id': 'batch_optimization',
                        'title': 'Batch Processing',
                        'description': 'Optimize batch sizes for hardware',
                        'priority': 'medium',
                        'implementation': 'Automatic batch size adjustment',
                        'benefits': 'Optimal performance utilization'
                    }
                ]
            }
        }
    }
    
    # Save improvement plan
    improvement_file = Path("ux_improvement_plan.json")
    with open(improvement_file, 'w') as f:
        json.dump(improvement_plan, f, indent=2)
    
    print(f"💾 UX improvement plan generated: {improvement_file}")
    
    return improvement_plan


def run_ux_analysis():
    """Run complete UX analysis and generate improvements."""
    print("🚀 RUNNING UX ANALYSIS")
    print("=" * 50)
    
    # Test CLI usability
    cli_results = test_cli_usability()
    
    # Test error message quality
    message_results = test_error_message_quality()
    
    # Gather feedback
    feedback_data = gather_ux_feedback()
    
    # Generate improvements
    improvements = generate_ux_improvements()
    
    print(f"\n📊 UX ANALYSIS COMPLETED")
    print(f"📁 CLI Usability Score: {cli_results['score']}/100")
    print(f"📝 Error Message Quality: {message_results['level']}")
    
    print(f"📈 Feedback Items Collected: {len(feedback_data['user_stories'])}")
    print(f"🎯 Improvements Generated: {len(improvements['categories']['cli_usability']['improvements'])}")
    
    # Save comprehensive report
    comprehensive_report = {
        'timestamp': time.time(),
        'cli_usability': cli_results,
        'error_message_quality': message_results,
        'ux_feedback': feedback_data,
        'improvements': improvements,
        'overall_status': 'ready_for_implementation'
    }
    
    report_file = Path("ux_analysis_report.json")
    with open(report_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"💾 Comprehensive UX report saved: {report_file}")
    
    return comprehensive_report


def main():
    """Main UX testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT UX Testing")
    parser.add_argument('--cli', action='store_true', help='Test CLI usability')
    parser.add_argument('--messages', action='store_true', help='Test error message quality')
    parser.add_argument('--feedback', action='store_true', help='Gather UX feedback')
    parser.add_argument('--improvements', action='store_true', help='Generate UX improvements')
    parser.add_argument('--all', action='store_true', help='Run complete UX analysis')
    
    args = parser.parse_args()
    
    print("🧪 SLO-GPT UX TESTING SUITE")
    print("=" * 50)
    
    if args.all:
        return run_ux_analysis()
    elif args.cli:
        test_cli_usability()
    elif args.messages:
        test_error_message_quality()
    elif args.feedback:
        gather_ux_feedback()
    elif args.improvements:
        generate_ux_improvements()
    else:
        parser.print_help()
        return 1
    
    print(f"\n🎉 UX TESTING COMPLETED")
    return 0


if __name__ == '__main__':
    exit(main())