"""
test_sentiment.py — Run with: python test_sentiment.py
Tests the accuracy of the sentiment engine on hand-labeled cases.
"""

from sentiment_engine import analyze

TEST_CASES = [
    # (text, expected_label, expected_emotion_hint)
    ("I absolutely love this product! It works perfectly every time.", "positive", "happy"),
    ("Thank you so much, you've been incredibly helpful!", "positive", "grateful"),
    ("This is amazing, best purchase I ever made!!!", "positive", "happy"),
    ("Great service, very fast delivery.", "positive", "default"),

    ("I'm really frustrated, this is the third time this has happened.", "negative", "frustrated"),
    ("This is absolutely terrible! My order never arrived and nobody helps!!", "negative", "angry"),
    ("I'm so disappointed. I expected much better quality.", "negative", "sad"),
    ("I'm really worried my package might be lost.", "negative", "anxious"),

    ("Can you tell me the shipping times?", "neutral", "default"),
    ("I need help resetting my password.", "neutral", "default"),
    ("I'm confused about the refund process.", "neutral", "confused"),
    ("What are your support hours?", "neutral", "default"),
]

PASS_COLOR = "\033[92m"
FAIL_COLOR = "\033[91m"
RESET = "\033[0m"

def run_tests():
    passed = 0
    print("\n── Sentiment Engine Test Results ──────────────────────────────\n")
    for text, expected_label, _ in TEST_CASES:
        result = analyze(text)
        ok = result.label == expected_label
        icon = "✅" if ok else "❌"
        color = PASS_COLOR if ok else FAIL_COLOR
        print(f"{color}{icon} [{expected_label.upper():8}] got [{result.label.upper():8}] | score {result.score:+.3f} | emotion: {result.emotion}{RESET}")
        print(f"   \"{text[:70]}\"")
        if ok:
            passed += 1

    total = len(TEST_CASES)
    accuracy = passed / total * 100
    print(f"\n── Results: {passed}/{total} passed ({accuracy:.1f}% accuracy) ──────────────\n")
    if accuracy >= 80:
        print(f"{PASS_COLOR}✅ PASS — meets 80% accuracy threshold{RESET}")
    else:
        print(f"{FAIL_COLOR}❌ FAIL — below 80% threshold{RESET}")

if __name__ == "__main__":
    run_tests()