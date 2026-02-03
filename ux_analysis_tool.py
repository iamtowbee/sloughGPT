#!/usr/bin/env python3
"""
Simple UX Analysis Tool for SloGPT
Focused analysis and user experience optimization.
"""

import os
import json
import time
from pathlib import Path

# Add system to path
sys.path.insert(0, str(Path(__file__).parent))


def analyze_current_system():
    """Analyze current SloGPT system for UX issues."""
    print("🔍 ANALYZING CURRENT SLO-GPT SYSTEM")
    print("=" * 50)
    
    # Check core files for common UX issues
    core_files = [
        'create_dataset_fixed.py',
        'train_simple.py',
        'huggingface_integration.py',
        'simple_distributed_training.py'
        'web_interface.py',
        'analytics_dashboard.py'
        'deployment_system.py'
    ]
    
    ux_issues = []
    
    for file_name in core_files:
        if not Path(file_name).exists():
            ux_issues.append({
                'file': file_name,
                'issue': 'missing_file',
                'severity': 'high',
                'description': f"Core file {file_name} not found"
            })
            continue
        
        print(f"📄 Checking: {file_name}")
        
        # Check for common CLI UX patterns
        try:
            with open(file_name, 'r') as f:
                content = f.read()
                
                # Look for common UX issues
                issues = []
                
                # Check for complex argument parsing
                if 'argparse.ArgumentParser' in content and 'add_argument' in content:
                    args_count = content.count('add_argument')
                    if args_count > 5:
                        issues.append({
                            'issue': 'too_many_arguments',
                            'severity': 'medium',
                            'description': f"{args_count} command line arguments may overwhelm users"
                        })
                
                # Check for complex help text
                if 'parser.print_help' in content and len(content.split('\n')) > 50:
                    issues.append({
                            'issue': 'verbose_help',
                            'severity': 'low',
                            'description': 'Help text is very long and verbose'
                        })
                
                # Check for cryptic error messages
                error_keywords = ['traceback', 'exception', 'error', 'failed']
                error_count = sum(1 for keyword in error_keywords if keyword in content.lower())
                
                if error_count > 3:
                    issues.append({
                            'issue': 'technical_error_messages',
                            'severity': 'medium',
                            'description': f"Error message contains {error_count} technical terms"
                        })
                
                # Check for missing examples
                if 'example' not in content.lower() and 'usage:' in content:
                    issues.append({
                            'issue': 'missing_examples',
                            'severity': 'medium',
                            'description': "No usage examples provided"
                        })
            
                if issues:
                    ux_issues.extend(issues)
                    print(f"   ⚠️ Found {len(issues)} potential UX issues")
                else:
                    print(f"   ✅ Good UX practices observed")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {file_name}: {e}")
            ux_issues.append({
                'file': file_name,
                'issue': 'analysis_error',
                'severity': 'high',
                'description': f"Failed to analyze: {e}"
            })
    
    # Generate recommendations
    recommendations = []
    
    # High priority fixes
    if any(issue['severity'] == 'high' for issue in ux_issues):
        recommendations.append("Create simplified CLI aliases for common workflows")
        recommendations.append("Improve error message clarity with actionable fixes")
        recommendations.append("Add comprehensive examples and tutorials")
    
    # Medium priority improvements
    medium_issues = [issue for issue in ux_issues if issue['severity'] == 'medium']
    if medium_issues:
        recommendations.append("Add in-context help system with --task flag")
        recommendations.append("Implement progressive disclosure in help system")
    
    # Low priority enhancements
    low_issues = [issue for issue in ux_issues if issue['severity'] == 'low']
    if low_issues:
        recommendations.append("Add interactive setup wizard for new users")
        recommendations.append("Implement user preference settings")
    
    if not recommendations:
        recommendations.append("Continue maintaining good UX practices")
    
    print(f"\n📊 ANALYSIS RESULTS")
    print(f"   Files Analyzed: {len(core_files)}")
    print(f"   Issues Found: {len(ux_issues)}")
    print(f"   Recommendations Generated: {len(recommendations)}")
    
    # Save analysis results
    analysis_results = {
        'timestamp': time.time(),
        'files_analyzed': core_files,
        'ux_issues': ux_issues,
        'recommendations': recommendations,
        'summary': {
            'total_issues': len(ux_issues),
            'high_priority_issues': len([i for i in ux_issues if i['severity'] == 'high']),
            'medium_priority_issues': len([i for i in ux_issues if i['severity'] == 'medium']),
            'low_priority_issues': len([i for i in ux_issues if i['severity'] == 'low']),
            'total_recommendations': len(recommendations)
        }
    }
    
    results_file = Path("ux_analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"💾 Analysis results saved: {results_file}")
    
    return analysis_results


def test_documentation_quality():
    """Test documentation quality and completeness."""
    print("📚 TESTING DOCUMENTATION QUALITY")
    print("=" * 50)
    
    doc_files = [
        'COMPLETE_USER_GUIDE.md',
        'DATASET_STANDARDIZATION.md', 
        'FINAL_COMPLETE_STATUS_REPORT.md',
        'HUGGINGFACE_INTEGRATION_COMPLETE.md'
    ]
    
    quality_metrics = {
        'total_files': len(doc_files),
        'missing_sections': [],
        'completeness_scores': [],
        'example_coverage': 0,
        'accessibility': 0
    }
    
    for doc_file in doc_files:
        if not Path(doc_file).exists():
            quality_metrics['missing_sections'].append(doc_file)
            print(f"   ❌ Missing: {doc_file}")
            continue
        
        print(f"📄 Checking: {doc_file}")
        
        try:
            with open(doc_file, 'r') as f:
                content = f.read()
                
                # Check for key sections
                key_sections = ['## ', '### ', 'Usage:', 'Installation:', 'Examples:', 'Troubleshooting:', 'FAQ', 'API', 'CLI']
                found_sections = [section for section in key_sections if section in content]
                quality_metrics['completeness_scores'].append(len(found_sections))
                
                # Check for examples
                example_count = content.count('```')
                quality_metrics['example_coverage'] += example_count
                
                # Check for accessibility
                if len(content) < 1000:
                    quality_metrics['accessibility'] += 1
                else:
                    quality_metrics['accessibility'] = 0
                
                print(f"   ✅ Found {len(found_sections)}/{len(key_sections)} sections")
                print(f"   📊 Examples: {example_count} code blocks")
            
        except Exception as e:
            print(f"   ❌ Error checking {doc_file}: {e}")
    
    # Calculate scores
    if quality_metrics['completeness_scores']:
        avg_completeness = sum(quality_metrics['completeness_scores']) / len(quality_metrics['completeness_scores'])
    else:
        avg_completeness = 0
    
    if quality_metrics['example_coverage'] > 0:
        example_score = min(100, quality_metrics['example_coverage'] / 10)
    else:
        example_score = 0
    
    overall_score = (avg_completeness + example_score + quality_metrics['accessibility']) / 3
    
    print(f"\n📚 DOCUMENTATION QUALITY RESULTS")
    print(f"   Files Analyzed: {quality_metrics['total_files']}")
    print(f"   Completeness: {avg_completeness:.1f}/100")
    print(f"   Example Coverage: {quality_metrics['example_coverage']} code blocks")
    print(f"   Accessibility: {quality_metrics['accessibility']}")
    print(f"   Overall Score: {overall_score:.1f}/100")
    
    # Save documentation quality results
    doc_results_file = Path("documentation_quality_results.json")
    with open(doc_results_file, 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    print(f"💾 Documentation quality results saved: {doc_results_file}")
    
    return quality_metrics


def main():
    """Main UX analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT UX Analysis Tool")
    parser.add_argument('--current', action='store_true', help='Analyze current system')
    parser.add_argument('--docs', action='store_true', help='Test documentation quality')
    parser.add_argument('--all', action='store_true', help='Run complete UX analysis')
    
    args = parser.parse_args()
    
    print("🧪 SLO-GPT UX ANALYSIS TOOL")
    print("=" * 50)
    
    if args.all:
        # Run all analyses
        system_results = analyze_current_system()
        doc_results = test_documentation_quality()
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS COMPLETED")
        print(f"📄 System Issues: {system_results['summary']['total_issues']}")
        print(f"📚 Documentation Score: {doc_results['overall_score']:.1f}/100}")
        
        # Combined score
        if system_results['summary']['total_issues'] == 0 and doc_results['overall_score'] > 80:
            print("🎉 EXCELLENT USER EXPERIENCE!")
        elif system_results['summary']['total_issues'] <= 5 and doc_results['overall_score'] > 70:
            print("✅ GOOD USER EXPERIENCE")
        else:
            print("⚠️ USER EXPERIENCE NEEDS IMPROVEMENT")
        
    elif args.docs:
        test_documentation_quality()
    elif args.current:
        analyze_current_system()
    else:
        parser.print_help()
        return 1
    
    print(f"\n🎉 ANALYSIS COMPLETED")
    return 0


if __name__ == '__main__':
    exit(main())