import unittest

from streamlit.testing.v1 import AppTest


def collect_text(at: AppTest) -> str:
    values = []
    for group_name in ("title", "header", "subheader", "caption", "markdown"):
        for element in getattr(at, group_name):
            values.append(str(getattr(element, "value", "")))
    return "\n".join(values)


class DashboardFeedbackTest(unittest.TestCase):
    def test_overview_and_sidebar_do_not_show_team_name(self):
        at = AppTest.from_file("dashboard/app.py")
        at.run()

        self.assertNotIn("Team kai", collect_text(at))

    def test_product_explorer_does_not_render_min_reviews_control(self):
        at = AppTest.from_file("dashboard/app.py")
        at.run()
        at.sidebar.radio[0].set_value("Product Explorer")
        at.run()

        number_input_labels = [control.label for control in at.number_input]

        self.assertNotIn("Min reviews", number_input_labels)


if __name__ == "__main__":
    unittest.main()
