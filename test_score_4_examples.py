#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.swt_service import evaluate_summary_service

def test_score_4_examples():
    """Test examples specifically designed to achieve content score 4"""
    
    test_cases = [
        {
            "name": "Example 1: Comprehensive Paraphrasing",
            "summary": "Global climate change presents humanity's most urgent challenge, with escalating temperatures from human-caused greenhouse gas emissions profoundly affecting worldwide ecosystems. Research has revealed substantial modifications in climatic patterns, ocean levels, and biological diversity across all continents. The Intergovernmental Panel on Climate Change cautions that failing to implement prompt emission controls will result in devastating outcomes such as intensified extreme weather phenomena, agricultural insecurity, and widespread human displacement. Nevertheless, scientific investigations have uncovered viable remedies including sustainable energy innovations, atmospheric carbon sequestration methods, and eco-friendly farming approaches.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 2: Excellent Synthesis with Connectors",
            "summary": "Humanity faces an unprecedented environmental crisis as climate change accelerates due to anthropogenic greenhouse gas emissions, consequently causing widespread ecological disruption. Scientific evidence demonstrates substantial alterations in meteorological conditions, coastal water levels, and species diversity throughout global regions. The IPCC emphasizes that delayed emission mitigation will precipitate disastrous ramifications encompassing intensified weather extremes, agricultural instability, and population migrations. However, investigators have discovered effective countermeasures including sustainable power generation, atmospheric carbon management, and environmentally conscious agricultural methodologies.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 3: Academic Style Summary",
            "summary": "Contemporary climate change constitutes humanity's paramount environmental challenge, characterized by escalating atmospheric temperatures resulting from anthropogenic greenhouse gas emissions that profoundly impact global ecosystems. Empirical research has documented substantial modifications in meteorological patterns, oceanic elevations, and biological diversity across continental regions. The Intergovernmental Panel on Climate Change has issued warnings regarding catastrophic consequences arising from delayed emission reduction initiatives, including intensified meteorological extremes, agricultural vulnerability, and demographic displacement. Conversely, scientific investigations have identified promising mitigation strategies encompassing sustainable energy technologies, carbon sequestration methodologies, and environmentally sustainable agricultural practices.",
            "reference": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are having profound effects on ecosystems worldwide. Scientists have documented significant changes in weather patterns, sea levels, and biodiversity loss across multiple continents. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce emissions, the world faces catastrophic consequences including more frequent extreme weather events, food insecurity, and mass displacement of populations. However, researchers have identified several promising solutions including renewable energy technologies, carbon capture and storage systems, and sustainable agricultural practices."
        },
        {
            "name": "Example 4: Concise but Comprehensive",
            "summary": "Climate change emerges as humanity's foremost environmental challenge, driven by human-generated greenhouse gas emissions that significantly impact global ecosystems. Scientific observations reveal substantial changes in weather systems, sea levels, and biodiversity across continents. The IPCC cautions that delayed emission reduction will cause catastrophic outcomes including extreme weather events, food shortages, and population displacement. However, researchers have identified effective solutions including renewable energy, carbon capture technology, and sustainable agriculture.",
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
            
            if result['content'] == 4:
                print(f"🎉 PERFECT: Achieved content score 4!")
            elif result['content'] >= 3:
                print(f"✅ GOOD: Got content score {result['content']}")
            else:
                print(f"⚠️  Content score is {result['content']}, needs improvement")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_score_4_examples()
