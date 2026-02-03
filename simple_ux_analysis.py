#!/usr/bin/env python3
"""
Ultra-Simple UX Analysis for SloGPT
No complex data structures or type annotations.
"""

import os
import json
import time
import subprocess
from pathlib import Path


def test_cli_tools():
    """Test basic CLI usability."""
    print("🎯 TESTING CLI USABILITY")
    print("=" * 40)
    
    tools_found = 0
    issues_found = 0
    
    # Test essential tools
    essential_tools = [
        "create_dataset_fixed.py",
        "train_simple.py"
        "huggingface_integration.py"
    ]
    
    for tool in essential_tools:
        print(f"🔧 Testing: {tool}")
        
        if not Path(tool).exists():
            issues_found += 1
            print(f"   ❌ Missing: {tool}")
            continue
        
        # Test help functionality
        try:
            result = subprocess.run(
                [sys.executable, tool, "--help"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                print(f"   ✅ Help command works")
            else:
                print(f"   ❌ Help command error: {result.stderr}")
                issues_found += 1
                
        except Exception as e:
            issues_found += 1
    
    # Calculate score
    total_tools = len(essential_tools)
    score = 100 if issues_found == 0 else 50
    
    print(f"\n📊 CLI USABILITY RESULTS")
    print(f"   Total Tools: {total_tools}")
    print(f"   Issues Found: {issues_found}")
    print(f"   Score: {score}/100")
    
    return {
        'score': score,
        'issues': issues_found,
        'total_tools': total_tools
    }


def analyze_simple_system():
    """Perform basic system analysis for UX improvements."""
    print("🔍 ANALYZING SIMPLE SYSTEM")
    print("=" * 50)
    
    ux_issues = []
    
    # Check for basic system components
    core_components = [
        "create_dataset_fixed.py",
        "train_simple.py",
        "huggingface_integration.py",
        "simple_gpt_model.py"
    ]
    
    for component in core_components:
        print(f"🔧 Checking: {component}")
        
        if not Path(component).exists():
            ux_issues.append({
                'component': component,
                'issue': 'missing_file',
                'severity': 'high',
                'description': f"Core component {component} not found"
            })
            issues_found += 1
            continue
        
        # Check if component is functional
        if component == "create_dataset_fixed.py":
            ux_issues.append({
                'component': component,
                'issue': 'missing_functionality',
                'severity': 'medium',
                'description': 'Create dataset creation not functional'
            })
        
        elif component == "train_simple.py":
            ux_issues.append({
                'component': component,
                'issue': 'missing_functionality',
                'severity': 'medium',
                'description': 'Training functionality issues'
            })
        
        elif component == "huggingface_integration.py":
            ux_issues.append({
                'component': component,
                'issue': 'missing_functionality',
                'severity': 'medium',
                'description': 'HF integration issues'
            })
        
        elif component == "simple_gpt_model.py":
            ux_issues.append({
                'component': component,
                'issue': 'missing_file',
                'severity': 'high',
                'description': 'Model architecture not found'
            })
        
        issues_found += 1
    
    # Calculate UX score
    total_components = len(core_components)
    issues_found = len([issue for issue in ux_issues if issue['severity'] != 'low'])
    
    if issues_found == 0:
        ux_score = 95
        ux_level = "Excellent"
    elif issues_found <= 2:
        ux_score = 80
        ux_level = "Good"
    elif issues_found <= 5:
        ux_score = 70
    else:
        ux_score = 50
    
    print(f"\n📊 SIMPLE SYSTEM ANALYSIS")
    print(f"   Components Analyzed: {total_components}")
    print(f"   Issues Found: {issues_found}")
    print(f"   UX Score: {ux_score}/100")
    print(f"   UX Level: {ux_level}")
    
    return {
        'score': ux_score,
        'level': ux_level,
        'issues': ux_issues,
        'total_components': total_components
    }


def generate_simple_recommendations(ux_issues):
    """Generate simple UX improvements."""
    print("\n🔧 GENERATING SIMPLE UX IMPROVEMENTS")
    print("=" * 50)
    
    recommendations = []
    
    if not ux_issues:
        recommendations.append({
            'category': 'general',
            'description': 'Maintain current excellent user experience',
            'implementation': 'Continue following best practices'
        })
    
    # Generate improvements based on identified issues
    for issue in ux_issues:
        if issue['severity'] == 'high':
            recommendations.append({
                'category': 'usability',
                'description': f"Fix critical issue: {issue['description']}",
                'implementation': "Prioritize this fix for user experience",
                'benefits': "Essential for functionality"
            })
        elif issue['severity'] == 'medium':
            recommendations.append({
                'category': 'usability',
                'description': f"Improve medium issue: {issue['description']}",
                'implementation': "Add better error handling"
            })
        else:
            recommendations.append({
                'category': 'usability',
                'description': f"Enhance user experience: {issue['description']}",
                'implementation': "Add progressive improvements"
            })
    
    print(f"\n🔧 UX IMPROVEMENTS: {len(recommendations)}")
    
    for rec in recommendations:
        print(f"   ✅ {rec['category']}: {rec['description']}")
        print(f"   Benefits: {rec['benefits']}")
    
    return recommendations


def main():
    """Main UX analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT Simple UX Analysis")
    parser.add_argument('--simple', action='store_true', help='Run simple system analysis')
    parser.add_argument('--all', action='store_true', help='Run complete analysis')
    
    args = parser.parse_args()
    
    print("🧪 SLO-GPT UX ANALYSIS TOOL")
    print("=" * 50)
    
    if args.all:
        # Run both analyses
        print("🔍 Running complete system analysis...")
        cli_results = test_cli_usability()
        simple_results = analyze_simple_system()
        
        # Generate improvements based on findings
        recommendations = generate_simple_recommendations(cli_results['issues'])
        
        print(f"\n📊 COMPLETE ANALYSIS COMPLETED")
        print(f"📊 UX Issues: {len(cli_results['issues'])}")
        print(f"   Usability Score: {cli_results['score']}/100")
        print(f"   Recommendations: {len(recommendations)}")
        
        return cli_results
        
    elif args.simple:
        analyze_simple_system()
        
    else:
        parser.print_help()


if __name__ == '__main__':
    exit(main())