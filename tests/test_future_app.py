import unittest
from datetime import date, time

from birth_chart import create_birth_chart
from future_app import create_app


class FutureAppTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_homepage_has_birth_input_and_past_life_option(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Welcome to the Future", page)
        self.assertIn('name="birth_date"', page)
        self.assertIn('name="birth_time"', page)
        self.assertIn('name="birth_location"', page)
        self.assertIn('name="future_focus"', page)
        self.assertIn("Future prediction", page)
        self.assertNotIn("Palm observation", page)
        self.assertIn("Past-life reflection", page)

    def test_production_health_and_security_headers(self):
        health = self.client.get("/healthz")
        homepage = self.client.get("/")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json(), {"status": "ok"})
        self.assertEqual(homepage.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(homepage.headers.get("X-Frame-Options"), "DENY")
        self.assertIn("frame-ancestors 'none'", homepage.headers.get("Content-Security-Policy", ""))

    def test_birth_reading_response_is_not_cached_or_persisted_in_a_cookie(self):
        response = self.client.post(
            "/reading",
            data={
                "birth_date": "1990-08-15",
                "birth_time": "09:30",
                "birth_location": "London, United Kingdom",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers.get("Cache-Control", ""))
        self.assertNotIn("Set-Cookie", response.headers)

    def test_public_form_limits_topic_length(self):
        response = self.client.post(
            "/reading",
            data={
                "birth_date": "1990-08-15",
                "birth_time": "09:30",
                "birth_location": "London, United Kingdom",
                "future_focus": "x" * 501,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Future prediction topic must be 500 characters or fewer", response.get_data(as_text=True))

    def test_reading_calculates_sun_sign_and_life_path(self):
        response = self.client.post(
            "/reading",
            data={
                "birth_date": "1990-08-15",
                "birth_time": "09:30",
                "birth_location": "London, United Kingdom",
                "future_focus": "career change",
            },
        )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Leo", page)
        self.assertIn("Life path 6", page)
        self.assertIn("career change", page)
        self.assertIn("one step within your control", page)
        self.assertIn("Professional life", page)
        self.assertIn("London, United Kingdom", page)
        self.assertIn("Chart placements use an offline city-center coordinate match", page)
        self.assertIn('class="workspace has-reading"', page)
        self.assertIn("not scientifically validated predictors", page)

    def test_past_life_is_presented_as_imaginative_reflection(self):
        response = self.client.post(
            "/past-life",
            data={
                "birth_date": "1990-08-15",
                "birth_time": "09:30",
                "birth_location": "London, United Kingdom",
            },
        )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Past-life reflection", page)
        self.assertIn("imaginative", page)

    def test_invalid_birth_date_returns_helpful_validation(self):
        response = self.client.post(
            "/reading",
            data={
                "birth_date": "not-a-date",
                "birth_time": "09:30",
                "birth_location": "London, United Kingdom",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Enter a valid birth date", response.get_data(as_text=True))

    def test_birth_details_create_chart_and_life_sections(self):
        response = self.client.post(
            "/reading",
            data={
                "birth_date": "1990-08-15",
                "birth_time": "09:30",
                "birth_location": "London, United Kingdom",
                "future_focus": "Tell me about my professional life",
            },
        )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        for item in ("Your birth chart", "Sun", "Moon", "Rising", "Midheaven", "Personal life", "Relationships", "Family and belonging", "Professional life"):
            with self.subTest(item=item):
                self.assertIn(item, page)
        self.assertIn('class="zodiac-wheel"', page)
        self.assertIn('data-chart-point="sun"', page)
        self.assertIn("London, United Kingdom", page)
        self.assertIn("does not determine your relationship status", page)

    def test_unknown_birthplace_is_not_silently_approximated(self):
        response = self.client.post(
            "/reading",
            data={
                "birth_date": "1990-08-15",
                "birth_time": "09:30",
                "birth_location": "Madeupville, Nowhere",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("could not match that city and country", response.get_data(as_text=True))

    def test_location_accepts_city_alias_and_region_component(self):
        chart = create_birth_chart(
            date(1990, 8, 15),
            time(9, 30),
            "New Delhi, Delhi, India",
        )

        self.assertEqual(chart["birthplace"], "Delhi, India")
        self.assertEqual(chart["timezone"], "Asia/Kolkata")

    def test_unmatched_city_suggests_supported_cities(self):
        response = self.client.post(
            "/reading",
            data={
                "birth_date": "1990-08-15",
                "birth_time": "09:30",
                "birth_location": "Madeupville, India",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Try a supported city in India", response.get_data(as_text=True))

    def test_birth_time_and_location_change_chart_angles(self):
        birth_date = date(1990, 8, 15)
        london_morning = create_birth_chart(birth_date, time(9, 30), "London, United Kingdom")
        london_evening = create_birth_chart(birth_date, time(21, 30), "London, United Kingdom")
        new_york = create_birth_chart(birth_date, time(9, 30), "New York, USA")

        self.assertNotEqual(
            london_morning["placements"]["Rising"]["sign"],
            london_evening["placements"]["Rising"]["sign"],
        )
        self.assertNotEqual(
            london_morning["placements"]["Rising"]["degree"],
            new_york["placements"]["Rising"]["degree"],
        )


if __name__ == "__main__":
    unittest.main()