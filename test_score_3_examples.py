#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.swt_service import evaluate_summary_service

def test_score_3_examples():
    """Test examples specifically designed to achieve content score 3"""
    
    test_cases = [
        {
            "name": "Example 1: Mixed Paraphrasing with Simple Connectors",
            "summary": "Climate change is a major problem for humanity. Global temperatures are rising because of greenhouse gas emissions from human activities, and this is affecting ecosystems around the world. Scientists have found significant changes in weather patterns, sea levels, and biodiversity loss across many continents. The IPCC warns that without immediate action to reduce emissions, there will be catastrophic consequences including extreme weather events, food insecurity, and mass displacement of people. However, researchers have found several promising solutions including renewable energy technologies, carbon capture systems, and sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 2: Good Coverage with Moderate Copying",
            "summary": "Climate change is one of the most important challenges facing humanity today. Global temperatures are rising because of greenhouse gas emissions from human activities, and this is having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The IPCC has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 3: Basic Paraphrasing with Some Repetition",
            "summary": "Climate change poses a significant threat to humanity. Rising temperatures caused by greenhouse gas emissions are impacting ecosystems globally. Scientists have observed important changes in weather patterns, sea levels, and biodiversity across continents. The IPCC has warned that without quick action to reduce emissions, there will be serious consequences including extreme weather events, food shortages, and population displacement. However, researchers have found several good solutions including renewable energy technologies, carbon capture systems, and sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 4: Inconsistent Paraphrasing",
            "summary": "Climate change is humanity's biggest environmental challenge. Temperatures are increasing due to greenhouse gas emissions from human activities, which affects ecosystems worldwide. Scientists have recorded major changes in weather patterns, sea levels, and biodiversity loss across continents. The IPCC has cautioned that without quick action to reduce emissions, the world will face terrible consequences including extreme weather events, food insecurity, and population displacement. However, researchers have discovered promising solutions including renewable energy technologies, carbon capture systems, and sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")
        
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
            
            if result['content'] == 3:
                print(f"🎉 PERFECT: Achieved content score 3!")
            elif result['content'] >= 3:
                print(f"✅ GOOD: Got content score {result['content']}")
            else:
                print(f"⚠️  Content score is {result['content']}, needs improvement")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_score_3_examples()
