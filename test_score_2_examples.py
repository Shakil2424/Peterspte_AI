#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.swt_service import evaluate_summary_service

def test_score_2_examples():
    """Test examples specifically designed to achieve content score 2"""
    
    test_cases = [
        {
            "name": "Example 1: Very Basic Paraphrasing",
            "summary": "Climate change is a big problem for humanity. Global temperatures are rising because of greenhouse gas emissions from human activities. This affects ecosystems worldwide. Scientists have found changes in weather patterns, sea levels, and biodiversity loss across continents. The IPCC says that without action to reduce emissions, there will be catastrophic consequences including extreme weather events, food insecurity, and mass displacement of populations. However, researchers have found solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 2: Poor Synthesis with Some Copying",
            "summary": "Climate change is happening. It is caused by greenhouse gas emissions from human activities. This affects ecosystems worldwide. Scientists have documented changes in weather patterns, sea levels, and biodiversity loss. The IPCC has warned about consequences. There are extreme weather events. There is food insecurity. There is mass displacement of populations. However, researchers have found solutions. They include renewable energy technologies. They include carbon capture and storage systems. They include sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 3: Mixed Important and Unimportant Details",
            "summary": "Climate change is a problem that affects many things. The weather is changing and temperatures are going up. Scientists have studied this and found that greenhouse gas emissions from human activities are causing global temperatures to rise. This is having effects on ecosystems worldwide. There are changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The IPCC has warned about catastrophic consequences. Some people might have to move because of food insecurity. However, researchers have found some solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 4: Basic Comprehension with Copying",
            "summary": "Climate change is a big problem. Global temperatures are rising because of greenhouse gas emissions from human activities. This affects ecosystems worldwide. Scientists have found changes in weather patterns, sea levels, and biodiversity loss across continents. The IPCC says that without action to reduce emissions, there will be catastrophic consequences including extreme weather events, food insecurity, and mass displacement of populations. However, researchers have found solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices.",
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
            
            if result['content'] == 2:
                print(f"🎉 PERFECT: Achieved content score 2!")
            elif result['content'] >= 2:
                print(f"✅ GOOD: Got content score {result['content']}")
            else:
                print(f"⚠️  Content score is {result['content']}, needs improvement")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_score_2_examples()
