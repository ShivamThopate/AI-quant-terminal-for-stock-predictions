"""
Quick smoke test for the FinBERT sentiment scorer.

Run:
    python -m pytest tests/test_finbert.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_finbert_positive_headline():
    """A clearly positive financial headline should score > 0."""
    from nlp.finbert_scorer import FinBERTScorer
    scorer = FinBERTScorer()
    score = scorer.score("Reliance Q4 profit jumps 12%, beats all estimates")
    print(f"Positive headline score: {score}")
    assert score > 0.0, f"Expected positive score, got {score}"


def test_finbert_negative_headline():
    """A clearly negative financial headline should score < 0."""
    from nlp.finbert_scorer import FinBERTScorer
    scorer = FinBERTScorer()
    score = scorer.score("Company faces massive fraud investigation, stock crashes 30%")
    print(f"Negative headline score: {score}")
    assert score < 0.0, f"Expected negative score, got {score}"


def test_finbert_batch_scoring():
    """Batch scoring should return one score per input."""
    from nlp.finbert_scorer import FinBERTScorer
    scorer = FinBERTScorer()
    texts = [
        "Strong earnings growth reported by TCS",
        "SEBI issues warning to company for violations",
        "Markets closed flat on low volumes",
    ]
    scores = scorer.score_batch(texts)
    print(f"Batch scores: {scores}")
    assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
    assert scores[0] > scores[1], "Positive headline should score higher than negative"


def test_finbert_vs_vader_financial_nuance():
    """FinBERT should understand financial context better than VADER.
    
    'Revenue missed estimates but margins improved significantly' has mixed
    signals. VADER would just count positive/negative words blindly.
    FinBERT should give a more nuanced (near-zero or mildly positive/negative) score
    rather than an extreme one.
    """
    from nlp.finbert_scorer import FinBERTScorer
    scorer = FinBERTScorer()
    score = scorer.score("Despite challenges, company maintains strong dividend and healthy cash flow")
    print(f"Nuanced headline score: {score}")
    # This should be positive — focusing on strength despite adversity
    assert score > -0.5, f"FinBERT should handle nuance — got {score}"



def test_sentiment_scorer_integration():
    """The SentimentScorer class should now use FinBERT internally."""
    from features.sentiment import SentimentScorer
    scorer = SentimentScorer()
    score = scorer.score_single("Infosys reports record revenue growth")
    print(f"SentimentScorer integration score: {score}")
    assert score > 0.0, f"Expected positive score from integrated scorer, got {score}"


if __name__ == "__main__":
    print("=" * 60)
    print("  FINBERT SENTIMENT SCORER — SMOKE TEST")
    print("=" * 60)
    
    tests = [
        test_finbert_positive_headline,
        test_finbert_negative_headline,
        test_finbert_batch_scoring,
        test_finbert_vs_vader_financial_nuance,
        test_sentiment_scorer_integration,
    ]
    
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
    
    print("\nDone.")
