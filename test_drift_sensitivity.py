#!/usr/bin/env python3
"""
Quick test to verify drift detection sensitivity improvements.
"""

import sys
import logging
from remgpt.detection.topic_drift_detector import TopicDriftDetector
from remgpt.core.types import UserMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_drift_sensitivity():
    """Test that related questions don't trigger false drift detection."""
    
    # Create detector with more conservative settings
    detector = TopicDriftDetector(
        similarity_threshold=0.6,
        drift_threshold=1.0,
        alpha=0.1,
        window_size=5
    )
    
    # Test conversation: Python programming topics
    messages = [
        "Can you explain what Python list comprehensions are?",
        "How do they compare to regular for loops?",  # Should NOT trigger drift
        "Tell me about lambda functions.",              # Should NOT trigger drift  
        "Tell me about Python decorators",            # Should NOT trigger drift
        "Can you give examples of decorators?",       # Should NOT trigger drift
        "What's the difference between supervised and unsupervised machine learning?",  # SHOULD trigger drift
        "What are some cooking techniques for pasta?", # SHOULD trigger drift
    ]
    
    print("🧪 Testing Drift Detection Sensitivity")
    print("=" * 50)
    
    results = []
    for i, message_text in enumerate(messages):
        message = UserMessage(content=message_text)
        drift_detected, _, similarity = detector.detect_drift(message)
        
        status = "🔴 DRIFT" if drift_detected else "🟢 SAME"
        print(f"{i+1}. {status} | {similarity:.3f} | {message_text}")
        
        results.append((message_text, drift_detected, similarity))
    
    # Analyze results
    print("\n📊 Analysis:")
    print("-" * 30)
    
    # These should NOT trigger drift (related Python topics)
    related_questions = [1, 2, 3, 4]  # indices of follow-up questions
    false_positives = 0
    
    for idx in related_questions:
        if results[idx][1]:  # if drift was detected
            false_positives += 1
            print(f"❌ False positive: '{results[idx][0][:40]}...' (similarity: {results[idx][2]:.3f})")
    
    # These should trigger drift (different topics)
    different_topics = [5, 6]  # indices of truly different topics
    true_positives = 0
    
    for idx in different_topics:
        if results[idx][1]:  # if drift was detected
            true_positives += 1
            print(f"✅ Correct detection: '{results[idx][0][:40]}...' (similarity: {results[idx][2]:.3f})")
    
    # Summary
    print(f"\n🎯 Results:")
    print(f"   False positives: {false_positives}/{len(related_questions)} (lower is better)")
    print(f"   True positives: {true_positives}/{len(different_topics)} (higher is better)")
    
    if false_positives == 0:
        print("🎉 SUCCESS: No false positives for related questions!")
    else:
        print(f"⚠️  Need improvement: {false_positives} false positives detected")
    
    return false_positives == 0

if __name__ == "__main__":
    success = test_drift_sensitivity()
    sys.exit(0 if success else 1) 