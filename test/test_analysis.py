import unittest
import pandas as pd
from src.sentiment_analysis import preprocess_for_nlp
from src.thematic_analysis import assign_themes

class TestNLPModule(unittest.TestCase):

    def test_preprocess_standard_text(self):
        """Test that text cleaning lowercases, tokens, and removes short/stop words."""
        sample_text = "The bank login app is incredibly slow and crashing!"
        result = preprocess_for_nlp(sample_text)
        
        # 'the', 'is' are stopwords. 'app' is length 3. 'slow', 'crashing' should remain.
        self.assertIn("slow", result)
        self.assertNotIn("the", result)

    def test_preprocess_empty_and_nan(self):
        """Test that missing values or empty strings return blank output without crashing."""
        self.assertEqual(preprocess_for_nlp(""), "")
        self.assertEqual(preprocess_for_nlp(pd.NA), "")

    def test_theme_assignment(self):
        """Test that specific technical keywords map to the correct business theme."""
        stability_text = "error crash slow network"
        access_text = "login otp password pin"
        
        self.assertIn("Performance & Stability", assign_themes(stability_text))
        self.assertIn("Account & Access", assign_themes(access_text))
        self.assertEqual(assign_themes("random words"), "General Feedback")

if __name__ == "__main__":
    unittest.main()