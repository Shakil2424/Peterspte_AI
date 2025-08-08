#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.swt_service import evaluate_summary_service

def test_improved_scoring():
    """Test the improved SWT scoring with various examples"""
    
    test_cases = [
        {
            "name": "Score 4 Example - Better Paraphrasing",
            "summary": "Global climate change represents a critical challenge for humanity, with rising temperatures from greenhouse gas emissions causing widespread ecosystem impacts. Scientists have observed significant alterations in weather patterns, sea levels, and biodiversity across continents. The IPCC warns that without immediate emission reductions, catastrophic consequences including extreme weather events, food insecurity, and population displacement will occur. However, researchers have identified promising solutions such as renewable energy technologies, carbon capture systems, and sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Score 3 Example - Good Coverage",
            "summary": "Climate change affects global ecosystems and requires sustainable solutions to address environmental challenges. Scientists have documented changes in weather patterns and biodiversity loss. The IPCC warns of catastrophic consequences without immediate action. Researchers have identified solutions including renewable energy and sustainable practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Score 2 Example - Basic Coverage",
            "summary": "Climate change is bad for the environment and we need to fix it with renewable energy.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*50}")
        
        try:
            result = evaluate_summary_service(test_case['summary'], test_case['reference'])
            
            print(f"✅ Content Score: {result['content']}/4")
            print(f"📊 Total Score: {result['total']}/9")
            
            if 'content_analysis' in result['details']:
                content_analysis = result['details']['content_analysis']
                print(f"\n📈 Content Analysis Details:")
                print(f"  - Similarity: {content_analysis.get('similarity', 0):.3f}")
                print(f"  - Idea Coverage: {content_analysis.get('idea_coverage', 0):.3f}")
                print(f"  - Paraphrasing Score: {content_analysis.get('paraphrasing_score', 0):.3f}")
                print(f"  - Connector Diversity: {content_analysis.get('connector_diversity', 0):.3f}")
                print(f"  - Synthesis Score: {content_analysis.get('synthesis_score', 0):.3f}")
                print(f"  - Copying Score: {content_analysis.get('copying_score', 0):.3f}")
                print(f"  - Rubric Level: {content_analysis.get('rubric_level', 'Unknown')}")
            
            # Check if we're getting higher scores now
            if result['content'] >= 3:
                print(f"🎉 SUCCESS: Got content score {result['content']} (improved!)")
            else:
                print(f"⚠️  Content score is {result['content']}, might need further adjustment")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_improved_scoring()
