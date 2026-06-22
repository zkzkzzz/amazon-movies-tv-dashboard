import unittest

from dashboard.app import predict_sentiment


class BrokenModel:
    def predict_proba(self, texts):
        raise RuntimeError("model is not usable")


class SentimentPredictionTest(unittest.TestCase):
    def test_prediction_errors_are_returned_instead_of_raised(self):
        result = predict_sentiment(BrokenModel(), "sample review")

        self.assertIsNone(result["label"])
        self.assertIsNone(result["positive_probability"])
        self.assertIn("model is not usable", result["error"])


if __name__ == "__main__":
    unittest.main()
