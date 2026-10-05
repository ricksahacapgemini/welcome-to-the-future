from __future__ import annotations

import math
import re
import unicodedata
from difflib import get_close_matches
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import geonamescache
from skyfield.api import load, load_file
from skyfield.framelib import ecliptic_frame
from skyfield_data import get_skyfield_data_path
from timezonefinder import TimezoneFinder


ZODIAC_SIGNS = (
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
)
EPHEMERIS_START = date(1900, 1, 1)
EPHEMERIS_END = date(2050, 12, 31)
_GEONAMES = geonamescache.GeonamesCache()
_COUNTRIES = _GEONAMES.get_countries()
_CITIES = tuple(_GEONAMES.get_cities().values())
_TIMEZONE_FINDER = TimezoneFinder(in_memory=True)
_EPHEMERIS = load_file(str(Path(get_skyfield_data_path()) / "de421.bsp"))
_TIMESCALE = load.timescale(builtin=True)
_PLANET_TARGETS = {
    "Sun": "sun",
    "Moon": "moon",
    "Mercury": "mercury",
    "Venus": "venus",
    "Mars": "mars",
    "Jupiter": "JUPITER BARYCENTER",
    "Saturn": "SATURN BARYCENTER",
    "Uranus": "URANUS BARYCENTER",
    "Neptune": "NEPTUNE BARYCENTER",
    "Pluto": "PLUTO BARYCENTER",
}


def _normalized_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_value.casefold())


def _country_code(country_name: str) -> str | None:
    normalized_country = _normalized_name(country_name)
    aliases = {
        "uk": "GB",
        "unitedkingdom": "GB",
        "greatbritain": "GB",
        "england": "GB",
        "scotland": "GB",
        "wales": "GB",
        "usa": "US",
        "unitedstates": "US",
        "unitedstatesofamerica": "US",
    }
    if normalized_country in aliases:
        return aliases[normalized_country]
    for country in _COUNTRIES.values():
        if normalized_country in {
            _normalized_name(str(country.get("name", ""))),
            _normalized_name(str(country.get("iso", ""))),
            _normalized_name(str(country.get("iso3", ""))),
        }:
            return str(country["iso"])
    return None


def _find_birthplace(location: str) -> dict[str, str | float]:
    location_parts = [part.strip() for part in location.split(",") if part.strip()]
    if len(location_parts) < 2:
        raise ValueError(
            "Enter birthplace as city, country. You may include a state or region, "
            "for example, New Delhi, Delhi, India."
        )

    city_parts = location_parts[:-1]
    country_name = location_parts[-1]
    requested_country_code = _country_code(country_name)
    if requested_country_code is None:
        raise ValueError(
            f"We could not match that city and country because '{country_name}' "
            "is not recognized in the offline location list."
        )

    city_aliases = {
        "newyork": "newyorkcity",
        "nyc": "newyorkcity",
        "newdelhi": "delhi",
        "bombay": "mumbai",
        "calcutta": "kolkata",
        "madras": "chennai",
    }
    normalized_candidates = []
    for city_part in city_parts:
        normalized_candidate = _normalized_name(city_part)
        normalized_candidates.append(city_aliases.get(normalized_candidate, normalized_candidate))

    country_cities = [
        city for city in _CITIES if city.get("countrycode") == requested_country_code
    ]
    city_index: dict[str, dict[str, object]] = {}
    for city in country_cities:
        normalized_name = _normalized_name(str(city.get("name", "")))
        current = city_index.get(normalized_name)
        if current is None or int(city.get("population", 0) or 0) > int(current.get("population", 0) or 0):
            city_index[normalized_name] = city

    matched_city = next(
        (city_index[candidate] for candidate in normalized_candidates if candidate in city_index),
        None,
    )
    if matched_city is None:
        country_display_name = str(_COUNTRIES[requested_country_code]["name"])
        name_by_normalized = {
            normalized_name: str(city["name"])
            for normalized_name, city in city_index.items()
        }
        close_matches = get_close_matches(
            normalized_candidates[0], list(name_by_normalized), n=3, cutoff=0.65
        )
        suggestions = [name_by_normalized[name] for name in close_matches]
        if not suggestions:
            popular_cities = sorted(
                city_index.values(),
                key=lambda city: int(city.get("population", 0) or 0),
                reverse=True,
            )
            suggestions = list(dict.fromkeys(str(city["name"]) for city in popular_cities[:8]))[:3]
        suggestion_text = ", ".join(suggestions)
        raise ValueError(
            "We could not match that city and country: "
            f"'{', '.join(city_parts)}' was not found in {country_display_name}'s offline city list. "
            f"Try a supported city in {country_display_name}, such as {suggestion_text}."
        )

    city = matched_city
    latitude = float(city["latitude"])
    longitude = float(city["longitude"])
    timezone_name = _TIMEZONE_FINDER.timezone_at(lat=latitude, lng=longitude)
    if timezone_name is None:
        raise ValueError("We could not determine a time zone for that birthplace.")

    country = _COUNTRIES[requested_country_code]
    return {
        "city": str(city["name"]),
        "country": str(country["name"]),
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
    }


def _sign_for_longitude(longitude: float) -> tuple[str, float]:
    normalized_longitude = longitude % 360
    sign_index = int(normalized_longitude // 30)
    return ZODIAC_SIGNS[sign_index], normalized_longitude % 30


def _obliquity_degrees(julian_date: float) -> float:
    centuries = (julian_date - 2451545.0) / 36525.0
    arcseconds = (
        84381.448
        - 46.8150 * centuries
        - 0.00059 * centuries**2
        + 0.001813 * centuries**3
    )
    return arcseconds / 3600


def _local_sidereal_degrees(skyfield_time, longitude: float) -> float:
    return (skyfield_time.gast * 15 + longitude) % 360


def _horizon_altitude(longitude: float, obliquity: float, latitude: float, sidereal_angle: float) -> tuple[float, float]:
    ecliptic_longitude = math.radians(longitude)
    obliquity_radians = math.radians(obliquity)
    latitude_radians = math.radians(latitude)
    right_ascension = math.atan2(
        math.sin(ecliptic_longitude) * math.cos(obliquity_radians),
        math.cos(ecliptic_longitude),
    )
    declination = math.asin(math.sin(ecliptic_longitude) * math.sin(obliquity_radians))
    hour_angle = (sidereal_angle - right_ascension + math.pi) % (2 * math.pi) - math.pi
    sine_altitude = (
        math.sin(latitude_radians) * math.sin(declination)
        + math.cos(latitude_radians) * math.cos(declination) * math.cos(hour_angle)
    )
    return sine_altitude, hour_angle


def _ascendant_longitude(obliquity: float, latitude: float, sidereal_degrees: float) -> float:
    sidereal_angle = math.radians(sidereal_degrees)
    sample_count = 720
    roots: list[float] = []
    previous_longitude = 0.0
    previous_altitude, _ = _horizon_altitude(
        previous_longitude, obliquity, latitude, sidereal_angle
    )

    for sample_index in range(1, sample_count + 1):
        current_longitude = sample_index * 360 / sample_count
        current_altitude, _ = _horizon_altitude(
            current_longitude, obliquity, latitude, sidereal_angle
        )
        if previous_altitude * current_altitude < 0:
            lower_bound = previous_longitude
            upper_bound = current_longitude
            lower_altitude = previous_altitude
            for _ in range(36):
                midpoint = (lower_bound + upper_bound) / 2
                midpoint_altitude, _ = _horizon_altitude(
                    midpoint % 360, obliquity, latitude, sidereal_angle
                )
                if lower_altitude * midpoint_altitude <= 0:
                    upper_bound = midpoint
                else:
                    lower_bound = midpoint
                    lower_altitude = midpoint_altitude
            root_longitude = ((lower_bound + upper_bound) / 2) % 360
            _, root_hour_angle = _horizon_altitude(
                root_longitude, obliquity, latitude, sidereal_angle
            )
            if root_hour_angle < 0:
                roots.append(root_longitude)
        previous_longitude = current_longitude
        previous_altitude = current_altitude

    if not roots:
        raise ValueError("Could not calculate a rising sign for that time and birthplace.")
    return roots[0]


def create_birth_chart(birth_date: date, birth_time: time, location: str) -> dict[str, object]:
    if not EPHEMERIS_START <= birth_date <= EPHEMERIS_END:
        raise ValueError("Chart calculations are available for birth dates from 1900 through 2050.")

    birthplace = _find_birthplace(location)
    local_datetime = datetime.combine(birth_date, birth_time).replace(
        tzinfo=ZoneInfo(str(birthplace["timezone"]))
    )
    utc_datetime = local_datetime.astimezone(timezone.utc)
    skyfield_time = _TIMESCALE.from_datetime(utc_datetime)
    observer = _EPHEMERIS["earth"].at(skyfield_time)

    placements: dict[str, dict[str, str | float]] = {}
    for planet_name, target_name in _PLANET_TARGETS.items():
        apparent_position = observer.observe(_EPHEMERIS[target_name]).apparent()
        ecliptic_longitude = apparent_position.frame_latlon(ecliptic_frame)[1].degrees
        sign, degree = _sign_for_longitude(ecliptic_longitude)
        placements[planet_name] = {
            "sign": sign,
            "degree": round(float(degree), 2),
            "longitude": round(float(ecliptic_longitude % 360), 2),
        }

    obliquity = _obliquity_degrees(skyfield_time.tt)
    sidereal_degrees = _local_sidereal_degrees(
        skyfield_time, float(birthplace["longitude"])
    )
    rising_longitude = _ascendant_longitude(
        obliquity, float(birthplace["latitude"]), sidereal_degrees
    )
    rising_sign, rising_degree = _sign_for_longitude(rising_longitude)
    midheaven_longitude = math.degrees(
        math.atan2(
            math.sin(math.radians(sidereal_degrees)),
            math.cos(math.radians(sidereal_degrees)) * math.cos(math.radians(obliquity)),
        )
    ) % 360
    midheaven_sign, midheaven_degree = _sign_for_longitude(midheaven_longitude)
    placements["Rising"] = {
        "sign": rising_sign,
        "degree": round(rising_degree, 2),
        "longitude": round(float(rising_longitude), 2),
    }
    placements["Midheaven"] = {
        "sign": midheaven_sign,
        "degree": round(midheaven_degree, 2),
        "longitude": round(float(midheaven_longitude), 2),
    }

    country = str(birthplace["country"])
    resolved_location = f"{birthplace['city']}, {country}"
    return {
        "birthplace": resolved_location,
        "timezone": str(birthplace["timezone"]),
        "coordinates": (float(birthplace["latitude"]), float(birthplace["longitude"])),
        "placements": placements,
    }