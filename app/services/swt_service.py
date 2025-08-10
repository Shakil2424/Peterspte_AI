from sentence_transformers import SentenceTransformer, util
from lexicalrichness import LexicalRichness
import language_tool_python
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import math

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Load models/tools once
sbert_model = SentenceTransformer('all-mpnet-base-v2', device="cpu")
lang_tool = language_tool_python.LanguageTool('en-US')

def extract_key_ideas(text):
    """Extract main ideas from text using sentence importance"""
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback to simple sentence splitting if NLTK fails
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentences = [s + '.' for s in sentences if not s.endswith('.')]
    
    if not sentences:
        return []
    
    # Encode all sentences
    sentence_embeddings = sbert_model.encode(sentences, convert_to_tensor=True)
    
    # Calculate importance based on similarity to other sentences
    importance_scores = []
    for i, emb in enumerate(sentence_embeddings):
        similarities = util.cos_sim(emb.unsqueeze(0), sentence_embeddings).squeeze()
        # Average similarity to other sentences (excluding self)
        avg_sim = (similarities.sum() - similarities[i]) / (len(similarities) - 1)
        importance_scores.append(avg_sim.item())
    
    # Get top sentences as key ideas
    top_indices = sorted(range(len(importance_scores)), key=lambda i: importance_scores[i], reverse=True)
    key_ideas = [sentences[i] for i in top_indices[:min(3, len(sentences))]]
    return key_ideas

def calculate_paraphrasing_score(summary, reference):
    """Calculate how well the summary paraphrases vs copies"""
    summary_words = set(word_tokenize(summary.lower()))
    reference_words = set(word_tokenize(reference.lower()))
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
    summary_words -= common_words
    reference_words -= common_words
    
    # Calculate overlap
    overlap = len(summary_words.intersection(reference_words))
    total_summary = len(summary_words)
    
    if total_summary == 0:
        return 0
    
    # Lower overlap = better paraphrasing
    paraphrasing_score = 1 - (overlap / total_summary)
    return paraphrasing_score

def calculate_connector_diversity(summary):
    """Calculate diversity of connective devices"""
    connectors = {
        'simple': ['and', 'but', 'or', 'so', 'because', 'when', 'if', 'then'],
        'complex': ['however', 'therefore', 'furthermore', 'moreover', 'nevertheless', 'consequently', 'in addition', 'on the other hand', 'for example', 'such as', 'in contrast', 'similarly', 'likewise', 'as a result', 'in conclusion', 'to summarize', 'including', 'specifically', 'particularly', 'especially', 'notably', 'significantly', 'importantly', 'additionally', 'moreover', 'furthermore', 'besides', 'also', 'as well as', 'along with', 'together with', 'in conjunction with', 'in combination with', 'in addition to', 'apart from', 'except for', 'other than', 'rather than', 'instead of', 'in place of', 'as an alternative to', 'as a substitute for', 'in lieu of', 'in the absence of', 'in spite of', 'despite', 'regardless of', 'irrespective of', 'notwithstanding', 'even though', 'although', 'though', 'while', 'whereas', 'on the contrary', 'by contrast', 'in comparison', 'compared to', 'compared with', 'in relation to', 'with respect to', 'regarding', 'concerning', 'as for', 'as to', 'in terms of', 'with regard to', 'in regard to', 'in reference to', 'in connection with', 'in association with', 'in collaboration with', 'in cooperation with', 'in partnership with', 'in alliance with', 'in conjunction with', 'in combination with', 'in coordination with', 'in synchronization with', 'in harmony with', 'in accordance with', 'in compliance with', 'in conformity with', 'in agreement with', 'in alignment with', 'in line with', 'in keeping with', 'in step with', 'in tune with', 'in sync with', 'in phase with', 'in parallel with', 'in tandem with', 'in concert with', 'in unison with', 'in solidarity with', 'in unity with', 'in collaboration with', 'in cooperation with', 'in partnership with', 'in alliance with', 'in league with', 'in cahoots with', 'in collusion with', 'in conspiracy with', 'in complicity with', 'in connivance with', 'in collusion with', 'in conspiracy with', 'in cahoots with', 'in league with', 'in alliance with', 'in partnership with', 'in cooperation with', 'in collaboration with', 'in conjunction with', 'in combination with', 'in coordination with', 'in synchronization with', 'in harmony with', 'in accordance with', 'in compliance with', 'in conformity with', 'in agreement with', 'in alignment with', 'in line with', 'in keeping with', 'in step with', 'in tune with', 'in sync with', 'in phase with', 'in parallel with', 'in tandem with', 'in concert with', 'in unison with', 'in solidarity with', 'in unity with']
    }
    
    words = word_tokenize(summary.lower())
    simple_count = sum(1 for word in words if word in connectors['simple'])
    complex_count = sum(1 for word in words if word in connectors['complex'])
    
    # Calculate diversity (prefer complex connectors)
    total_connectors = simple_count + complex_count
    if total_connectors == 0:
        return 0
    
    # More generous scoring for connector diversity
    if complex_count > 0:
        # Give bonus for using complex connectors
        diversity_score = min(2.0, (complex_count / total_connectors) * 2 + 0.8)
    else:
        # Even simple connectors get some points
        diversity_score = min(1.0, (simple_count / max(len(words), 1)) * 1.5)
    
    return diversity_score

def evaluate_summary_service(summary, reference):
    scores = {}
    
    # === 1. Form (1 point) ===
    words = summary.strip().split()
    word_count = len(words)
    sentence_count = len(re.findall(r'[.!?]', summary.strip()))
    if 5 <= word_count <= 75 and sentence_count == 1 and not summary.isupper():
        scores['form'] = 1
    else:
        scores['form'] = 0
    
    # === 2. Content (4 points) - Updated rubric ===
    content_result = evaluate_content_comprehension(summary, reference)
    scores['content'] = content_result['score']
    
    # === 3. Grammar (2 points) ===
    matches = lang_tool.check(summary)
    grammar_errors = [m for m in matches if m.ruleIssueType in ("grammar", "typographical")]
    spelling_errors = [m for m in matches if m.ruleIssueType in ("spelling", "misspelling")]
    num_errors = len(grammar_errors)
    if num_errors == 0:
        scores['grammar'] = 2
    elif num_errors <= 2:
        scores['grammar'] = 1
    else:
        scores['grammar'] = 0
    
    # === 4. Vocabulary (2 points) ===
    lex = LexicalRichness(summary)
    ttr = lex.ttr
    mtld = lex.mtld()
    if ttr > 0.7 and mtld > 20:
        scores['vocabulary'] = 2
    elif ttr > 0.5 and mtld > 15:
        scores['vocabulary'] = 1
    else:
        scores['vocabulary'] = 0
    
    # === Word Highlights for Grammar/Typo/Spelling Errors ===
    word_highlights = []
    for word in words:
        word_highlights.append({"word": word, "status": "correct", "replacement": None})
    
    # Process grammar errors
    for error in grammar_errors:
        error_text = summary[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        for i, highlight in enumerate(word_highlights):
            # Clean the highlight word (remove punctuation)
            clean_highlight = highlight['word'].rstrip('.,!?;:')
            if clean_highlight in error_words or highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "grammar",
                    "replacement": error.replacements[0] if error.replacements else None,
                }
                # Don't break - highlight all matching words
    
    # Process spelling errors (show in word highlights but don't affect score)
    for error in spelling_errors:
        error_text = summary[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        suggestions = error.replacements[:3] if error.replacements else []
        for i, highlight in enumerate(word_highlights):
            # Clean the highlight word (remove punctuation)
            clean_highlight = highlight['word'].rstrip('.,!?;:')
            if clean_highlight in error_words or highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "spelling",
                    "suggestions": suggestions,
                }
                # Don't break - highlight all matching words
    
    # === Final Total ===
    total = sum(scores.values())
    scores['total'] = total
    
    # Add score out of 90
    scores['score'] = math.ceil((total / 9) * 90)
    
    # Add details for API response
    scores['details'] = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'grammar_errors': [err.message for err in grammar_errors],
        'spelling_errors': [err.message for err in spelling_errors],
        'ttr': ttr,
        'mtld': mtld,
        'content_analysis': content_result
    }
    scores['word_highlights'] = word_highlights
    return scores

def evaluate_content_comprehension(summary, reference):
    """Evaluate content comprehension using the exact 4-point rubric"""
    
    # 1. Semantic similarity
    emb_ref = sbert_model.encode(reference, convert_to_tensor=True)
    emb_sum = sbert_model.encode(summary, convert_to_tensor=True)
    similarity = util.cos_sim(emb_ref, emb_sum).item()
    
    # 2. Key idea coverage
    key_ideas = extract_key_ideas(reference)
    try:
        summary_sentences = sent_tokenize(summary)
    except LookupError:
        # Fallback to simple sentence splitting if NLTK fails
        summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
        summary_sentences = [s + '.' for s in summary_sentences if not s.endswith('.')]
    
    idea_coverage = 0
    if key_ideas and summary_sentences:
        summary_emb = sbert_model.encode(summary_sentences, convert_to_tensor=True)
        key_emb = sbert_model.encode(key_ideas, convert_to_tensor=True)
        
        # Calculate how many key ideas are covered
        sim_matrix = util.cos_sim(key_emb, summary_emb)
        max_sims = sim_matrix.max(dim=1).values
        covered_ideas = sum(1 for sim in max_sims if sim.item() > 0.6)
        idea_coverage = covered_ideas / len(key_ideas) if key_ideas else 0
    
    # 3. Paraphrasing quality
    paraphrasing_score = calculate_paraphrasing_score(summary, reference)
    
    # 4. Connector diversity
    connector_diversity = calculate_connector_diversity(summary)
    
    # 5. Coherence (sentence similarity within summary)
    coherence_score = 0
    if len(summary_sentences) > 1:
        summary_emb = sbert_model.encode(summary_sentences, convert_to_tensor=True)
        sim_matrix = util.cos_sim(summary_emb, summary_emb)
        # Average similarity between sentences (excluding diagonal)
        coherence_score = ((sim_matrix.sum() - sim_matrix.trace()) / (sim_matrix.numel() - sim_matrix.size(0))).item()
    
    # 6. Detail removal assessment (check if summary is concise)
    reference_words = len(reference.split())
    summary_words = len(summary.split())
    conciseness_ratio = summary_words / reference_words if reference_words > 0 else 1
    
    # 7. Synthesis quality (how well ideas are combined vs copied)
    synthesis_score = calculate_synthesis_quality(summary, reference)
    
    # 8. Copying detection (how much is directly copied)
    copying_score = calculate_copying_score(summary, reference)
    
    # Scoring logic based on the exact rubric criteria
    score = 0
    
    # 4 points - Full comprehension & synthesis
    if (similarity >= 0.60 and 
        idea_coverage >= 0.60 and 
        paraphrasing_score >= 0.2 and 
        connector_diversity >= 0.3 and 
        coherence_score >= 0.05 and
        conciseness_ratio <= 0.8 and
        synthesis_score >= 0.2 and
        copying_score <= 0.7):
        score = 4
    
    # 3 points - Good comprehension
    elif (similarity >= 0.50 and 
          idea_coverage >= 0.45 and 
          paraphrasing_score >= 0.15 and 
          connector_diversity >= 0.1 and
          synthesis_score >= 0.1 and
          copying_score <= 0.8):
        score = 3
    
    # 2 points - Basic comprehension
    elif (similarity >= 0.50 and 
          idea_coverage >= 0.35 and 
          paraphrasing_score >= 0.15 and
          synthesis_score >= 0.15 and
          copying_score <= 0.8):
        score = 2
    
    # 1 point - Limited comprehension
    elif (similarity >= 0.35 and 
          idea_coverage >= 0.20 and
          synthesis_score >= 0.1 and
          copying_score <= 0.85):
        score = 1
    
    # 0 points - No comprehension
    else:
        score = 0
    
    return {
        'score': score,
        'similarity': similarity,
        'idea_coverage': idea_coverage,
        'paraphrasing_score': paraphrasing_score,
        'connector_diversity': connector_diversity,
        'coherence_score': coherence_score,
        'conciseness_ratio': conciseness_ratio,
        'synthesis_score': synthesis_score,
        'copying_score': copying_score,
        'key_ideas_found': len(key_ideas),
        'summary_sentences': len(summary_sentences),
        'rubric_level': get_rubric_description(score)
    }

def calculate_synthesis_quality(summary, reference):
    """Calculate how well the summary synthesizes vs copies"""
    summary_words = set(word_tokenize(summary.lower()))
    reference_words = set(word_tokenize(reference.lower()))
    
    # Remove common words and short words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
    summary_content_words = {w for w in summary_words if w not in common_words and len(w) > 3}
    reference_content_words = {w for w in reference_words if w not in common_words and len(w) > 3}
    
    if not summary_content_words:
        return 0
    
    # Calculate unique content words (synthesis)
    unique_content_words = summary_content_words - reference_content_words
    synthesis_ratio = len(unique_content_words) / len(summary_content_words)
    
    # Boost score for good paraphrasing (calculate paraphrasing score here)
    paraphrasing_score = calculate_paraphrasing_score(summary, reference)
    
    # More generous synthesis scoring
    if paraphrasing_score >= 0.4:
        synthesis_ratio = min(1.0, synthesis_ratio + 0.4)
    elif paraphrasing_score >= 0.2:
        synthesis_ratio = min(1.0, synthesis_ratio + 0.3)
    
    # Additional boost for good vocabulary diversity
    if len(summary_content_words) >= 6:
        synthesis_ratio = min(1.0, synthesis_ratio + 0.2)
    
    # Bonus for longer summaries with good vocabulary
    if len(summary_content_words) >= 8:
        synthesis_ratio = min(1.0, synthesis_ratio + 0.1)
    
    return synthesis_ratio

def calculate_copying_score(summary, reference):
    """Calculate how much of the summary is directly copied from reference"""
    summary_words = set(word_tokenize(summary.lower()))
    reference_words = set(word_tokenize(reference.lower()))
    
    # Remove common words and short words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'she', 'his', 'her', 'him', 'i', 'me', 'my', 'as', 'so', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose'}
    summary_content_words = summary_words - common_words
    reference_content_words = reference_words - common_words
    
    if not summary_content_words:
        return 0
    
    # Only count words longer than 4 characters to avoid false positives
    summary_content_words = {w for w in summary_content_words if len(w) > 4}
    reference_content_words = {w for w in reference_content_words if len(w) > 4}
    
    if not summary_content_words:
        return 0
    
    overlap = len(summary_content_words.intersection(reference_content_words))
    copying_ratio = overlap / len(summary_content_words)
    
    # More lenient copying detection - reduce the score
    return copying_ratio * 0.6

def get_rubric_description(score):
    """Get the rubric description for the score"""
    descriptions = {
        4: "Full comprehension & synthesis - Covers all main ideas from the source text, paraphrases effectively (not just copying sentences), removes unnecessary details, presents ideas concisely, clearly, and coherently, uses varied and appropriate connective devices",
        3: "Good comprehension - Captures the main ideas, but may omit some minor points, paraphrasing is present but inconsistent; some repetition from the text, ideas are connected, but synthesis could be better, uses mostly simple or repetitive connectors",
        2: "Basic comprehension - Includes some main ideas but mixes in less important details, relies heavily on copying phrases from the text, lacks synthesis and reformulation, connectors may be repetitive or poorly used",
        1: "Limited comprehension - Relevant to the source but not a real summary, disconnected ideas or chunks copied from the text, omits or misrepresents key points, lacks coherence",
        0: "No comprehension - Too limited to score; does not convey the main content of the source"
    }
    return descriptions.get(score, "Unknown score level") 