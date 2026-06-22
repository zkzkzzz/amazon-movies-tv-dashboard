import unittest

import pandas as pd

from dashboard.app import prepare_ranked_category_chart


class CategoryDisplayTest(unittest.TestCase):
    def test_excludes_unspecified_labels_and_reports_hidden_volume(self):
        categories = pd.DataFrame(
            {
                "brand": ["Unknown", "Various", ".", "Known A", "Known B"],
                "n_reviews": [500, 80, 20, 120, 90],
                "avg_rating": [4.1, 4.2, 4.3, 4.4, 4.5],
            }
        )

        chart, hidden_reviews = prepare_ranked_category_chart(categories, "brand", top_n=10)

        self.assertEqual(chart["brand"].tolist(), ["Known B", "Known A"])
        self.assertEqual(hidden_reviews, 600)


if __name__ == "__main__":
    unittest.main()
