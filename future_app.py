from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from flask import Flask, render_template, request

from birth_chart import create_birth_chart


SIGN_REFLECTIONS = {
    "Aries": "A useful prompt: where could a little patience make your next bold move stronger?",
    "Taurus": "A useful prompt: which steady habit is giving you more than quick wins would?",
    "Gemini": "A useful prompt: which conversation deserves your full attention rather than a quick reply?",
    "Cancer": "A useful prompt: how can you care for others while protecting time for yourself?",
    "Leo": "A useful prompt: where could you share the spotlight and make a good idea travel further?",
    "Virgo": "A useful prompt: what would be good enough to finish today, instead of perfect someday?",
    "Libra": "A useful prompt: which choice feels fair to you as well as to everyone around you?",
    "Scorpio": "A useful prompt: what might change if you asked one more honest question?",
    "Sagittarius": "A useful prompt: what new perspective could you explore without rushing past the details?",
    "Capricorn": "A useful prompt: which long-term goal can you advance with one manageable step?",
    "Aquarius": "A useful prompt: how could an unconventional idea become useful to the people around you?",
    "Pisces": "A useful prompt: how can you give your imagination room while staying grounded in what you know?",
}

NUMBER_REFLECTIONS = {
    1: "Independence and initiative: notice where you can begin without waiting for perfect conditions.",
    2: "Cooperation and balance: consider what a thoughtful partnership could make easier.",
    3: "Expression and curiosity: make space for a creative idea or an honest conversation.",
    4: "Structure and patience: small, repeatable steps may feel more useful than a dramatic reset.",
    5: "Change and exploration: stay open to options while keeping one or two steady anchors.",
    6: "Care and responsibility: offer support without taking on more than is yours to carry.",
    7: "Reflection and learning: give yourself quiet time to examine an assumption or follow a question.",
    8: "Planning and agency: define what success means to you before measuring progress.",
    9: "Perspective and completion: consider what is ready to be finished or shared with others.",
}

CAREER_SIGN_THEMES = {
    "Aries": "initiative and taking the first step",
    "Taurus": "steady execution and building durable value",
    "Gemini": "communication, learning, and connecting ideas",
    "Cancer": "supportive leadership and understanding people",
    "Leo": "creative ownership and visible contribution",
    "Virgo": "careful analysis and improving how work gets done",
    "Libra": "collaboration, negotiation, and fair decisions",
    "Scorpio": "focused research and navigating complex change",
    "Sagittarius": "teaching, exploration, and big-picture thinking",
    "Capricorn": "long-range planning and dependable progress",
    "Aquarius": "independent thinking and systems-level ideas",
    "Pisces": "imagination, empathy, and creative problem-solving",
}

CAREER_NUMBER_THEMES = {
    1: "initiative",
    2: "cooperation",
    3: "communication",
    4: "organization",
    5: "adaptability",
    6: "service and responsibility",
    7: "research and reflection",
    8: "planning and stewardship",
    9: "purpose and perspective",
}

PAST_LIFE_REFLECTIONS = {
    1: "Imagine a life shaped by learning to lead a small community through a difficult change. What would you carry forward: courage, or the wisdom to listen first?",
    2: "Imagine a life spent building bridges between people who rarely agreed. Which skill would you bring into your life today: diplomacy, or clearer boundaries?",
    3: "Imagine a life filled with stories, music, or teaching. What would you want to express now that might have gone unsaid then?",
    4: "Imagine a life devoted to patient craft and steady work. What would you build today if you trusted that small efforts accumulate?",
    5: "Imagine a life of travel and unexpected turns. Which kind of freedom matters most to you now, and what helps you use it well?",
    6: "Imagine a life centered on tending a home or caring for a close-knit group. How might you balance generosity with care for yourself?",
    7: "Imagine a life spent studying nature, ideas, or old books. What question are you ready to explore with fresh eyes?",
    8: "Imagine a life organizing resources during a time of change. How would you use influence responsibly in the present?",
    9: "Imagine a life of service that reached beyond one community. What unfinished kindness or creative work would you choose to continue?",
}


def _life_path_number(birth_date: date) -> int:
    digit_total = sum(int(digit) for digit in birth_date.strftime("%Y%m%d"))
    while digit_total > 9:
        digit_total = sum(int(digit) for digit in str(digit_total))
    return digit_total


def _read_profile(form: Any) -> tuple[dict[str, Any] | None, str | None]:
    for field, limit, label in (
        ("name", 80, "Name"),
        ("birth_location", 180, "Birthplace"),
        ("future_focus", 500, "Future prediction topic"),
    ):
        if len((form.get(field) or "").strip()) > limit:
            return None, f"{label} must be {limit} characters or fewer."

    raw_date = (form.get("birth_date") or "").strip()
    raw_time = (form.get("birth_time") or "").strip()
    try:
        birth_date = date.fromisoformat(raw_date)
    except ValueError:
        return None, "Enter a valid birth date."

    if birth_date > date.today():
        return None, "Birth date cannot be in the future."

    try:
        parsed_birth_time = datetime.strptime(raw_time, "%H:%M").time()
    except ValueError:
        return None, "Enter a valid birth time."

    birth_location = (form.get("birth_location") or "").strip()
    if not birth_location:
        return None, "Enter your birthplace (city and country)."
    try:
        chart = create_birth_chart(birth_date, parsed_birth_time, birth_location)
    except ValueError as error:
        return None, str(error)

    return {
        "name": (form.get("name") or "").strip(),
        "birth_date": birth_date.isoformat(),
        "birth_time": parsed_birth_time.strftime("%H:%M"),
        "birth_location": chart["birthplace"],
        "timezone": chart["timezone"],
        "chart": chart,
        "future_focus": (form.get("future_focus") or "").strip(),
    }, None


def _make_reading(profile: dict[str, Any]) -> dict[str, Any]:
    birth_date = date.fromisoformat(profile["birth_date"])
    placements = profile["chart"]["placements"]
    sign = str(placements["Sun"]["sign"])
    moon_sign = str(placements["Moon"]["sign"])
    rising_sign = str(placements["Rising"]["sign"])
    venus_sign = str(placements["Venus"]["sign"])
    midheaven_sign = str(placements["Midheaven"]["sign"])
    saturn_sign = str(placements["Saturn"]["sign"])
    life_path = _life_path_number(birth_date)
    future_focus = profile["future_focus"]
    focus_words = future_focus.casefold()
    career_focus = any(word in focus_words for word in ("career", "professional", "work", "job", "business"))
    career_analysis = (
        f"Your Midheaven in {midheaven_sign} is traditionally associated with "
        f"{CAREER_SIGN_THEMES[midheaven_sign]}, while Saturn in {saturn_sign} adds a symbolic "
        f"lens of {CAREER_SIGN_THEMES[saturn_sign]}. Numerology life path {life_path} is linked "
        f"with {CAREER_NUMBER_THEMES[life_path]}. Reflect on which of these themes fits your "
        "experience, which skill you want to strengthen, and whether your current direction fits "
        "your values. This does not measure aptitude or forecast a career outcome."
    )
    personal_analysis = (
        f"Your Sun in {sign} and Rising sign in {rising_sign} are symbolic lenses for identity "
        "and how you approach new situations. Consider which parts feel familiar and which do not; "
        "a chart cannot define who you are."
    )
    relationship_analysis = (
        f"Venus in {venus_sign} and Moon in {moon_sign} are traditionally used as prompts about "
        "affection, communication, and emotional needs. Use them to reflect on what helps you "
        "feel respected and understood. A chart does not determine your relationship status, "
        "compatibility, or future partners."
    )
    family_analysis = (
        f"Moon in {moon_sign} can be used as a symbolic prompt about comfort, care, and belonging. "
        "Think about the kinds of support and boundaries that matter to you. A chart does not "
        "determine your family status, history, or future."
    )
    if future_focus:
        future_reflection = (
            "You asked about: “"
            + future_focus
            + "” Birth details cannot determine what will happen. For this area, consider one step "
            "within your control, one uncertainty to check, and one person or resource that could help."
        )
    else:
        future_reflection = (
            "Choose an area you are thinking about, then consider one step within your control, "
            "one uncertainty to check, and one person or resource that could help."
        )
    return {
        "sign": sign,
        "life_path": life_path,
        "birth_time": profile["birth_time"],
        "birth_location": profile["birth_location"],
        "timezone": profile["timezone"],
        "placements": placements,
        "wheel_signs": [
            {
                "label": label,
                "x": round(250 + 185 * math.cos(math.radians(index * 30 + 15 - 90)), 2),
                "y": round(250 + 185 * math.sin(math.radians(index * 30 + 15 - 90)), 2),
            }
            for index, label in enumerate(
                ("ARI", "TAU", "GEM", "CAN", "LEO", "VIR", "LIB", "SCO", "SAG", "CAP", "AQU", "PIS")
            )
        ],
        "wheel_points": [
            {
                "name": name,
                "sign": str(placement["sign"]),
                "degree": float(placement["degree"]),
                "longitude": float(placement.get("longitude", 0.0)),
                "x": round(250 + 174 * math.cos(math.radians(float(placement.get("longitude", 0.0)) - 90)), 2),
                "y": round(250 + 174 * math.sin(math.radians(float(placement.get("longitude", 0.0)) - 90)), 2),
            }
            for name, placement in placements.items()
        ],
        "career_focus": career_focus,
        "career_analysis": career_analysis,
        "personal_analysis": personal_analysis,
        "relationship_analysis": relationship_analysis,
        "family_analysis": family_analysis,
        "location_notice": (
            "Chart placements use an offline city-center coordinate match and the local time zone. "
            "If the city match is approximate, the Rising sign and Midheaven may differ."
        ),
        "astrology_reflection": SIGN_REFLECTIONS[sign],
        "number_reflection": NUMBER_REFLECTIONS[life_path],
        "future_reflection": future_reflection,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

    @app.after_request
    def add_response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "form-action 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        if request.endpoint in {"reading", "past_life"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/healthz", methods=["GET"])
    def health():
        return {"status": "ok"}

    @app.route("/", methods=["GET"])
    def index():
        return render_template("future.html")

    @app.route("/reading", methods=["POST"])
    def reading():
        profile, error = _read_profile(request.form)
        if error:
            return render_template("future.html", error=error, form=request.form), 400
        return render_template(
            "future.html",
            profile=profile,
            reading=_make_reading(profile),
        )

    @app.route("/past-life", methods=["POST"])
    def past_life():
        profile, error = _read_profile(request.form)
        if error:
            return render_template("future.html", error=error, form=request.form), 400
        result = _make_reading(profile)
        result["past_life"] = PAST_LIFE_REFLECTIONS[result["life_path"]]
        return render_template("future.html", profile=profile, reading=result, show_past_life=True)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=False)