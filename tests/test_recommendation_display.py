import unittest

import pandas as pd

from dashboard.app import prepare_recommendation_display


class RecommendationDisplayTest(unittest.TestCase):
    def test_clamps_displayed_als_scores_without_changing_raw_predictions(self):
        recommendations = pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "asin": ["A", "B", "C"],
                "title": ["High", "Low", "In range"],
                "predicted_rating": [5.81, 0.2, 4.4],
            }
        )

        display = prepare_recommendation_display(recommendations)

        self.assertEqual(display["display_rating"].tolist(), [5.0, 1.0, 4.4])
        self.assertEqual(display["predicted_rating"].tolist(), [5.81, 0.2, 4.4])


if __name__ == "__main__":
    unittest.main()
