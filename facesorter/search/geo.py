"""
facesorter/search/geo.py
Offline gazetteer: GPS coordinate -> ``PlaceLabel`` and place name -> ``PlaceFilter``.

Place data comes from the ``geonamescache`` package (GeoNames, https://www.geonames.org,
licensed CC BY 4.0; the package itself is MIT). Nothing is downloaded and no data file is
bundled with facesorter. Attribution: "Place data (c) GeoNames, CC BY 4.0".

Labelling rule (``label_for`` / ``label_many``)
-----------------------------------------------
GeoNames lists real cities next to thousands of neighbourhoods, boroughs and districts
("Chinatown", "Paris 16 Passy", "Parkside", "Manhattan"). Snapping a photo to the nearest
point therefore gives neighbourhood names, so labelling works in three steps.

1. *Footprint.* Every place gets a footprint radius that grows with its population:
   ``R = clamp(0.25 * sqrt(population / 1000), 2, 40)`` km (San Jose ~7.9 km, New York
   City ~23 km, a 15 000-person town 2 km).
2. *Labelable set.* A place is labelable when its population is at least ``min_population``
   (default 15 000) and it is not a neighbourhood of a bigger place in the same country and
   first-level region. Going from the largest place down, ``N`` is absorbed by an
   unabsorbed ``C`` when either
   (a) ``N`` is at least 10x smaller than ``C`` and lies within ``R(C)`` (the outskirts: any
   much smaller place there is a district or suburb-sized neighbourhood), or
   (b) ``N`` is at least 3x smaller than ``C`` and lies within the core of ``C``,
   ``0.6 * R(C)`` (rising to ``0.85 * R(C)`` for cities of 5 million or more), so boroughs
   and big districts such as Manhattan, Brooklyn, Queens belong to New York City.
   Composite district names ("Diamond Head / Kapahulu", "Zürich (Kreis 11)", "Paris 16
   Passy") are never labelable; real places do not carry slashes, brackets or digits.
   Two cities of similar size stay separate (Sunnyvale, Milpitas, Mountain View, Santa Clara
   and San Jose are all labelable), as do places in a different region (Hoboken, NJ is not
   part of New York City).
3. *Assignment.* A coordinate is given to a labelable place it is plausibly "in": within
   ``max(12 km, 2 * R)`` and at most ``max_city_km`` (default 50 km). Among those, places
   whose own footprint contains the point come first (nearest wins); otherwise the smallest
   ``distance / R``. Normalising by footprint lets a big city claim its outskirts while a
   small town's own centre still wins over a big city 6 km away. All places within
   ``max_city_km`` are considered (a fast k-nearest query, redone as a radius query when
   the k nearest are all inside city range, as in dense metros).

If no city qualifies, the label falls back to the country (and region when they agree):
the nearest labelable place within ``max_country_km`` (default 250 km) names the country
if every place of another country is at least 15% farther, so remote spots (a national park,
a mountain village) are still "Switzerland" or "Arizona, United States"; points close to a
border, and ocean, polar or other points with nothing that close, give ``None``.

Known limitations of the labelling rule: GeoNames has no feature codes in this data, so
metropolitan satellites of mega-cities may merge into them (Guarulhos and Osasco into
Sao Paulo, Boulogne-Billancourt into Paris); a neighbourhood listed with a population
close to its city's (Kampung Larkin Lama beside Johor Bahru) stays a separate label; and
cities facing each other across a river or border within ~2 km (Windsor and Detroit) are
decided by distance to the GeoNames point, which can be on the wrong side.

Region names
------------
GeoNames gives only first-level region *codes* (``admin1``), and ``geonamescache`` ships
names only for the 50 US states plus DC. This module adds built-in code tables for Canada,
Australia, the United Kingdom nations, India, Germany, Italy, France (current regions),
Spain (autonomous communities) and Austria. Every code is checked in the tests against a
well-known city; codes that could not be verified are omitted. For every other country
(Japan, China, Brazil, Mexico, ...) ``region`` is ``None``, so a query such as "Hokkaido" is
not understood unless it is a notable place (below). Region abbreviations ("BC", "NSW", "TX")
are accepted as qualifiers ("Vancouver, BC") and, in capitals, on their own.

Name resolution (``resolve`` / ``candidates``)
----------------------------------------------
Names are matched case-, diacritic- and punctuation-insensitively ("Zurich" == "Zürich",
"St. Louis" == "Saint Louis"). Accepted forms: ``City``, ``City, Region``, ``City, Country``
(also without the comma when the tail is a known region or country), a state or province,
a country (names, ISO codes written in capitals such as ``FR``/``USA``, and aliases such as
``UK``, ``Holland`` or ``Cote d'Ivoire``), aliases for cities (``NYC``, ``SF``, ``LA``,
``DC``, ``New York`` -> New York City; former names and exonyms such as Calcutta, Bombay,
Peking, Cracow) and a curated table of well-known destinations that are not cities
(``_NOTABLE_PLACES``: Cornwall, Bali, Lake Tahoe, Yosemite, ... -> ``near`` filters). City
matches use GeoNames primary names plus the ASCII alternate names of places with at least
15 000 people (alternates shorter than four characters are ignored to avoid words like
"dog"). An alternate name of a place at least 20 times bigger than the primary-name match
outranks it (Rio -> Rio de Janeiro, not the village of Réo).

Ambiguous names are ranked by a fixed, documented rule: curated destination or city alias,
then country, then region, then city (primary names before alternate names, then larger
population first, then country code and name for determinism). So "Georgia" is the country
(the US state is the next candidate), "Washington" is the state (then Washington, DC),
"Victoria" the Australian state, "Paris" is Paris, France, "Amherst" is Amherst, NY
(122 000) ahead of Amherst, MA (40 000). ``candidates`` returns the full ranked list;
qualifiers ("Paris, Texas", "Amherst, MA", "Georgia, USA") narrow it. A qualifier that is not
understood makes the whole name unresolved, and unknown names give ``None``: this module
never guesses. Names that exist only as tiny namesake towns elsewhere (a curated place like
"Interlaken" excepted) can still resolve to those towns; the curated table covers the
destinations people actually photograph.

Data note: results depend on the installed ``geonamescache`` data (populations, names and
codes change between releases); tests use tolerant thresholds for that reason.

``resolve`` output is chosen so that store filters agree with ``label_for``:

* labelable city -> ``PlaceFilter(city=, region=, country=)`` with the exact spellings
  that ``label_for`` emits;
* a real place that is not labelable (a neighbourhood, or a town below ``min_population``
  such as Whistler) -> ``PlaceFilter(near=(lat, lon, radius_km))``, because photos there
  are labelled with a nearby bigger city. Its data comes lazily from the GeoNames 5 000+
  list the first time a name is not found among the main places (about 0.5 s once);
* region -> ``region=`` and ``country=``; country -> ``country=``.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import ModelUnavailableError
from .types import PlaceFilter, PlaceLabel

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_MIN_POPULATION", "OfflineGazetteer", "build_gazetteer"]

DEFAULT_MIN_POPULATION = 15_000

_EARTH_KM = 6371.0088
_FOOTPRINT_K = 0.25  # km per sqrt(thousand people)
_FOOTPRINT_MIN_KM = 2.0
_FOOTPRINT_MAX_KM = 40.0
_OUTSKIRTS_RATIO = 10.0  # absorbed within R(C) when this many times smaller
_CORE_RATIO = 3.0  # absorbed within _CORE_FRACTION * R(C) when this many times smaller
_CORE_FRACTION = 0.6  # for cities up to _CORE_GROWTH_FROM people ...
_CORE_FRACTION_MAX = 0.85  # ... growing to this for cities of _CORE_GROWTH_TO or more
_CORE_GROWTH_FROM = 1_000_000
_CORE_GROWTH_TO = 5_000_000
_REACH_MIN_KM = 12.0
_REACH_FACTOR = 2.0
_CITY_STATES = frozenset({"HK", "SG", "MO"})  # admin1 there is a district, not a region
_COUNTRY_MARGIN = 0.85  # nearest country must be this much closer than any other
_REGION_MARGIN = 1.5  # region kept if all same-country places this close agree
_NEAREST_K = 16  # neighbours fetched per coordinate; dense areas fall back to a radius query
_OUTSIDE_FOOTPRINT = 1e6  # sort offset: places not containing the point rank after those that do
_NOTABLE_REGION_MARGIN_KM = 120.0  # region qualifiers match places this far around a destination
_NICKNAME_RATIO = 20.0  # an alternate-name match this many times bigger beats the primary name
_ALT_MIN_POPULATION = 15_000  # alternate names are indexed only for places this big
_ALT_MIN_LENGTH = 4
_NEAR_MIN_KM = 5.0  # radius of a ``near`` filter is at least this
_MAX_CANDIDATES = 12
_MAX_NAME_LENGTH = 80
_MAX_PARTS = 4  # comma-separated pieces of a name that are considered
_GEONAMES_FILES = (500, 1000, 5000, 15000)  # city lists shipped by geonamescache
_WIDE_FILE = 5000

# GeoNames admin1 codes -> names. US states come from geonamescache itself.
_CA_PROVINCES = {
    "01": "Alberta",
    "02": "British Columbia",
    "03": "Manitoba",
    "04": "New Brunswick",
    "05": "Newfoundland and Labrador",
    "07": "Nova Scotia",
    "08": "Ontario",
    "09": "Prince Edward Island",
    "10": "Quebec",
    "11": "Saskatchewan",
    "12": "Yukon",
    "13": "Northwest Territories",
    "14": "Nunavut",
}
_AU_STATES = {
    "01": "Australian Capital Territory",
    "02": "New South Wales",
    "03": "Northern Territory",
    "04": "Queensland",
    "05": "South Australia",
    "06": "Tasmania",
    "07": "Victoria",
    "08": "Western Australia",
}
_GB_NATIONS = {
    "ENG": "England",
    "SCT": "Scotland",
    "WLS": "Wales",
    "NIR": "Northern Ireland",
}
# India: GeoNames codes verified against well-known cities in the tests. Lakshadweep (no city
# in the data) is left out.
_IN_STATES = {
    "01": "Andaman and Nicobar Islands",
    "02": "Andhra Pradesh",
    "03": "Assam",
    "05": "Chandigarh",
    "07": "Delhi",
    "09": "Gujarat",
    "10": "Haryana",
    "11": "Himachal Pradesh",
    "12": "Jammu and Kashmir",
    "13": "Kerala",
    "16": "Maharashtra",
    "17": "Manipur",
    "18": "Meghalaya",
    "19": "Karnataka",
    "20": "Nagaland",
    "21": "Odisha",
    "22": "Puducherry",
    "23": "Punjab",
    "24": "Rajasthan",
    "25": "Tamil Nadu",
    "26": "Tripura",
    "28": "West Bengal",
    "29": "Sikkim",
    "30": "Arunachal Pradesh",
    "31": "Mizoram",
    "33": "Goa",
    "34": "Bihar",
    "35": "Madhya Pradesh",
    "36": "Uttar Pradesh",
    "37": "Chhattisgarh",
    "38": "Jharkhand",
    "39": "Uttarakhand",
    "40": "Telangana",
    "41": "Ladakh",
    "52": "Dadra and Nagar Haveli and Daman and Diu",
}
_DE_STATES = {
    "01": "Baden-Württemberg",
    "02": "Bavaria",
    "03": "Bremen",
    "04": "Hamburg",
    "05": "Hesse",
    "06": "Lower Saxony",
    "07": "North Rhine-Westphalia",
    "08": "Rhineland-Palatinate",
    "09": "Saarland",
    "10": "Schleswig-Holstein",
    "11": "Brandenburg",
    "12": "Mecklenburg-Vorpommern",
    "13": "Saxony",
    "14": "Saxony-Anhalt",
    "15": "Thuringia",
    "16": "Berlin",
}
_IT_REGIONS = {
    "01": "Abruzzo",
    "02": "Basilicata",
    "03": "Calabria",
    "04": "Campania",
    "05": "Emilia-Romagna",
    "06": "Friuli-Venezia Giulia",
    "07": "Lazio",
    "08": "Liguria",
    "09": "Lombardy",
    "10": "Marche",
    "11": "Molise",
    "12": "Piedmont",
    "13": "Apulia",
    "14": "Sardinia",
    "15": "Sicily",
    "16": "Tuscany",
    "17": "Trentino-Alto Adige",
    "18": "Umbria",
    "19": "Aosta Valley",
    "20": "Veneto",
}
_FR_REGIONS = {
    "11": "Île-de-France",
    "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté",
    "28": "Normandy",
    "32": "Hauts-de-France",
    "44": "Grand Est",
    "52": "Pays de la Loire",
    "53": "Brittany",
    "75": "Nouvelle-Aquitaine",
    "76": "Occitanie",
    "84": "Auvergne-Rhône-Alpes",
    "93": "Provence-Alpes-Côte d'Azur",
    "94": "Corsica",
}
_ES_REGIONS = {
    "07": "Balearic Islands",
    "27": "La Rioja",
    "29": "Community of Madrid",
    "31": "Murcia",
    "32": "Navarre",
    "34": "Asturias",
    "39": "Cantabria",
    "51": "Andalusia",
    "52": "Aragon",
    "53": "Canary Islands",
    "54": "Castilla-La Mancha",
    "55": "Castile and León",
    "56": "Catalonia",
    "57": "Extremadura",
    "58": "Galicia",
    "59": "Basque Country",
    "60": "Valencian Community",
}
_AT_STATES = {
    "02": "Carinthia",
    "03": "Lower Austria",
    "04": "Upper Austria",
    "05": "Salzburg",
    "06": "Styria",
    "07": "Tyrol",
    "08": "Vorarlberg",
    "09": "Vienna",
}
# Other countries (Japan, China, Brazil, Mexico, ...) have region=None: not verified.
_STATIC_REGIONS = {
    "CA": _CA_PROVINCES,
    "AU": _AU_STATES,
    "GB": _GB_NATIONS,
    "IN": _IN_STATES,
    "DE": _DE_STATES,
    "IT": _IT_REGIONS,
    "FR": _FR_REGIONS,
    "ES": _ES_REGIONS,
    "AT": _AT_STATES,
}

# Written abbreviations of regions (US state codes are added from geonamescache).
_REGION_ABBREVIATIONS = {
    "AB": [("CA", "01")],
    "BC": [("CA", "02")],
    "MB": [("CA", "03")],
    "NB": [("CA", "04")],
    "NL": [("CA", "05")],
    "NS": [("CA", "07")],
    "ON": [("CA", "08")],
    "PE": [("CA", "09")],
    "QC": [("CA", "10")],
    "SK": [("CA", "11")],
    "YT": [("CA", "12")],
    "NT": [("CA", "13"), ("AU", "03")],
    "NU": [("CA", "14")],
    "ACT": [("AU", "01")],
    "NSW": [("AU", "02")],
    "QLD": [("AU", "04")],
    "SA": [("AU", "05")],
    "TAS": [("AU", "06")],
    "VIC": [("AU", "07")],
    "WA": [("AU", "08")],
}

# Other spellings of region names: alias -> (country code, region code).
_REGION_ALIASES = {
    "Kashmir": ("IN", "12"),
    "Himachal": ("IN", "11"),
    "Orissa": ("IN", "21"),
    "Tamilnadu": ("IN", "25"),
    "Bayern": ("DE", "02"),
    "Hessen": ("DE", "05"),
    "Niedersachsen": ("DE", "06"),
    "Nordrhein-Westfalen": ("DE", "07"),
    "Sachsen": ("DE", "13"),
    "Thüringen": ("DE", "15"),
    "Toscana": ("IT", "16"),
    "Lombardia": ("IT", "09"),
    "Piemonte": ("IT", "12"),
    "Sicilia": ("IT", "15"),
    "Sardegna": ("IT", "14"),
    "Puglia": ("IT", "13"),
    "Friuli": ("IT", "06"),
    "Trentino": ("IT", "17"),
    "South Tyrol": ("IT", "17"),
    "Alto Adige": ("IT", "17"),
    "Andalucía": ("ES", "51"),
    "Cataluña": ("ES", "56"),
    "Catalunya": ("ES", "56"),
    "Pais Vasco": ("ES", "59"),
    "Basque": ("ES", "59"),
    "Provence": ("FR", "93"),
    "Normandie": ("FR", "28"),
    "Bretagne": ("FR", "53"),
    "Corse": ("FR", "94"),
    "Tirol": ("AT", "07"),
    "Steiermark": ("AT", "06"),
    "Kärnten": ("AT", "02"),
}

# Well-known destinations that are not (or cannot be labelled as) a region or a big city:
# (names, country codes, lat, lon, radius_km). They resolve to ``near`` filters and take
# precedence over small namesake towns ("Interlaken, California"). Radii are conservative.
_NOTABLE_PLACES = [
    # Great Britain
    (("Cornwall",), ("GB",), 50.35, -4.85, 45.0),
    (("Yorkshire",), ("GB",), 53.95, -1.3, 60.0),
    (("Lake District", "the Lakes"), ("GB",), 54.45, -3.05, 30.0),
    (("Cotswolds",), ("GB",), 51.85, -1.8, 40.0),
    (("Peak District",), ("GB",), 53.35, -1.8, 30.0),
    (("Snowdonia",), ("GB",), 53.07, -3.9, 35.0),
    (("Scottish Highlands", "the Highlands", "Highlands"), ("GB",), 57.2, -4.8, 90.0),
    (("Isle of Skye", "Skye"), ("GB",), 57.3, -6.2, 35.0),
    (("Loch Ness",), ("GB",), 57.32, -4.45, 20.0),
    # Southern and western Europe
    (("Algarve",), ("PT",), 37.1, -8.1, 70.0),
    (("Madeira",), ("PT",), 32.75, -16.97, 30.0),
    (("Mallorca", "Majorca"), ("ES",), 39.6, 2.95, 45.0),
    (("Ibiza",), ("ES",), 38.98, 1.43, 20.0),
    (("Tenerife",), ("ES",), 28.27, -16.6, 35.0),
    (("Costa del Sol",), ("ES",), 36.55, -4.6, 60.0),
    (("Santorini",), ("GR",), 36.39, 25.46, 15.0),
    (("Mykonos",), ("GR",), 37.45, 25.33, 12.0),
    (("Crete",), ("GR",), 35.24, 24.81, 90.0),
    (("Amalfi Coast",), ("IT",), 40.63, 14.6, 25.0),
    (("Cinque Terre",), ("IT",), 44.12, 9.71, 12.0),
    (("Lake Como",), ("IT",), 46.0, 9.26, 30.0),
    (("Lake Garda",), ("IT",), 45.65, 10.65, 25.0),
    (("Dolomites",), ("IT",), 46.4, 11.85, 60.0),
    (("French Riviera", "Cote d'Azur", "Riviera"), ("FR",), 43.7, 7.2, 60.0),
    (("Alsace",), ("FR",), 48.3, 7.5, 60.0),
    (("Dordogne",), ("FR",), 44.9, 0.9, 60.0),
    (("Loire Valley",), ("FR",), 47.4, 0.7, 90.0),
    (("Black Forest",), ("DE",), 48.1, 8.2, 60.0),
    (("Interlaken",), ("CH",), 46.6863, 7.8632, 12.0),
    (("Zermatt",), ("CH",), 46.0207, 7.7491, 10.0),
    # Asia, Africa, Oceania, South America
    (("Bali",), ("ID",), -8.4, 115.19, 60.0),
    (("Angkor Wat", "Angkor"), ("KH",), 13.4125, 103.867, 25.0),
    (("Mount Fuji", "Fuji"), ("JP",), 35.36, 138.73, 25.0),
    (("Mount Everest", "Everest"), ("NP", "CN"), 27.99, 86.93, 40.0),
    (("Kilimanjaro",), ("TZ",), -3.07, 37.36, 35.0),
    (("Serengeti",), ("TZ",), -2.33, 34.83, 100.0),
    (("Petra",), ("JO",), 30.33, 35.44, 10.0),
    (("Uluru", "Ayers Rock"), ("AU",), -25.34, 131.04, 25.0),
    (("Machu Picchu",), ("PE",), -13.1631, -72.545, 20.0),
    (("Patagonia",), ("AR", "CL"), -47.0, -71.0, 500.0),
    # United States and Canada
    (("Lake Tahoe", "Tahoe"), ("US",), 39.0968, -120.0324, 25.0),
    (("Yosemite",), ("US",), 37.8651, -119.5383, 40.0),
    (("Yellowstone",), ("US",), 44.46, -110.83, 70.0),
    (("Grand Canyon",), ("US",), 36.1, -112.1, 60.0),
    (("Zion",), ("US",), 37.2982, -113.0263, 30.0),
    (("Joshua Tree",), ("US",), 33.87, -115.9, 35.0),
    (("Death Valley",), ("US",), 36.5, -117.1, 60.0),
    (("Monument Valley",), ("US",), 36.998, -110.0985, 25.0),
    (("Napa Valley",), ("US",), 38.45, -122.35, 25.0),
    (("Big Sur",), ("US",), 36.27, -121.81, 40.0),
    (("Mount Rainier", "Rainier"), ("US",), 46.8523, -121.7603, 30.0),
    (("Olympic Peninsula",), ("US",), 47.8, -123.6, 70.0),
    (("San Juan Islands",), ("US",), 48.55, -123.05, 30.0),
    (("Glacier National Park",), ("US",), 48.7596, -113.787, 60.0),
    (("Cape Cod",), ("US",), 41.67, -70.2, 45.0),
    (("the Hamptons", "Hamptons"), ("US",), 40.96, -72.3, 30.0),
    (("Martha's Vineyard",), ("US",), 41.39, -70.62, 20.0),
    (("Nantucket",), ("US",), 41.28, -70.1, 15.0),
    (("Outer Banks",), ("US",), 35.6, -75.5, 70.0),
    (("Florida Keys", "the Keys"), ("US",), 24.85, -80.9, 100.0),
    (("White Mountains",), ("US",), 44.1, -71.5, 50.0),
    (("Adirondacks",), ("US",), 44.0, -74.2, 100.0),
    (("Catskills",), ("US",), 42.1, -74.4, 50.0),
    (("Berkshires",), ("US",), 42.35, -73.2, 40.0),
    (("Finger Lakes",), ("US",), 42.7, -76.8, 70.0),
    (("Maui",), ("US",), 20.8, -156.33, 40.0),
    (("Oahu",), ("US",), 21.47, -157.98, 35.0),
    (("Big Island",), ("US",), 19.6, -155.5, 80.0),
    (("Kauai",), ("US",), 22.05, -159.5, 30.0),
    (("Banff",), ("CA",), 51.1784, -115.5708, 30.0),
    (("Lake Louise",), ("CA",), 51.4254, -116.18, 15.0),
    (("Jasper",), ("CA",), 52.8737, -118.0814, 40.0),
    # Landmarks (small radii)
    (("Eiffel Tower",), ("FR",), 48.8584, 2.2945, 1.5),
    (("Big Ben",), ("GB",), 51.5007, -0.1246, 1.0),
    (("Colosseum",), ("IT",), 41.8902, 12.4922, 1.0),
    (("Sagrada Familia",), ("ES",), 41.4036, 2.1744, 1.0),
    (("Taj Mahal",), ("IN",), 27.1751, 78.0421, 1.5),
    (("Sydney Opera House",), ("AU",), -33.8568, 151.2153, 1.0),
    (("Statue of Liberty",), ("US",), 40.6892, -74.0445, 1.5),
    (("Golden Gate Bridge",), ("US",), 37.8199, -122.4783, 3.0),
    (("Times Square",), ("US",), 40.758, -73.9855, 1.0),
]

# City aliases (nicknames, exonyms, former names): alias -> (GeoNames name, country code).
# These win over small namesake towns ("Rio" is not Réo). Keys are normalised at import.
_CITY_ALIASES_RAW = {
    "NYC": ("New York City", "US"),
    "New York": ("New York City", "US"),
    "New York City": ("New York City", "US"),
    "LA": ("Los Angeles", "US"),
    "SF": ("San Francisco", "US"),
    "San Fran": ("San Francisco", "US"),
    "DC": ("Washington", "US"),
    "Washington DC": ("Washington", "US"),
    "Philly": ("Philadelphia", "US"),
    "Vegas": ("Las Vegas", "US"),
    "NOLA": ("New Orleans", "US"),
    "Rio": ("Rio de Janeiro", "BR"),
    "Calcutta": ("Kolkata", "IN"),
    "Bombay": ("Mumbai", "IN"),
    "Madras": ("Chennai", "IN"),
    "Bangalore": ("Bengaluru", "IN"),
    "Peking": ("Beijing", "CN"),
    "Saigon": ("Ho Chi Minh City", "VN"),
    "Cracow": ("Kraków", "PL"),
    "Munich": ("Munich", "DE"),
    "Naples": ("Naples", "IT"),
    "Florence": ("Florence", "IT"),
    "Venice": ("Venice", "IT"),
    "Rome": ("Rome", "IT"),
    "Vienna": ("Vienna", "AT"),
}

# Country aliases: normalised alias -> ISO 3166 alpha-2 code.
_COUNTRY_ALIASES_RAW = {
    "usa": "US",
    "u s a": "US",
    "united states of america": "US",
    "america": "US",
    "uk": "GB",
    "u k": "GB",
    "britain": "GB",
    "great britain": "GB",
    "uae": "AE",
    "holland": "NL",
    "netherlands": "NL",
    "czech republic": "CZ",
    "korea": "KR",
    "republic of korea": "KR",
    "burma": "MM",
    "côte d'ivoire": "CI",
    "cote d ivoire": "CI",
    "russian federation": "RU",
    "turkiye": "TR",
    "macau": "MO",
    "east timor": "TL",
    "vatican city": "VA",
    "viet nam": "VN",
    "swaziland": "SZ",
    "palestine": "PS",
}

_FOLD = str.maketrans(
    {"ø": "o", "đ": "d", "ł": "l", "æ": "ae", "œ": "oe", "ı": "i", "ð": "d", "þ": "th"}
)
_PUNCT = str.maketrans({"-": " ", "_": " ", ".": None, "'": None, "’": None, "`": None})
_ABBREVIATIONS = {"st": "saint", "ste": "sainte", "mt": "mount", "ft": "fort"}

_RANK_ALIAS, _RANK_COUNTRY, _RANK_REGION, _RANK_CITY, _RANK_CITY_ALT = 0, 1, 2, 3, 4


def _norm(text: str) -> str:
    """Fold case, diacritics and punctuation: ``"St. Zürich-Nord"`` -> ``"saint zurich nord"``."""
    t = text.casefold()
    if not t.isascii():
        t = unicodedata.normalize("NFKD", t).translate(_FOLD)
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return " ".join(_ABBREVIATIONS.get(w, w) for w in t.translate(_PUNCT).split())


_CITY_ALIASES = {_norm(k): v for k, v in _CITY_ALIASES_RAW.items()}
_COUNTRY_ALIASES = {_norm(k): v for k, v in _COUNTRY_ALIASES_RAW.items()}
_REGION_ALIAS_INDEX = {_norm(k): v for k, v in _REGION_ALIASES.items()}


def _is_composite(name: str) -> bool:
    """GeoNames district names like "Diamond Head / Kapahulu" or "Zürich (Kreis 11)"."""
    return " / " in name or "(" in name or any(ch.isdigit() for ch in name)


def _core_fraction(population: float) -> float:
    """Share of a footprint that counts as the city's core: wider for megacities."""
    growth = np.log(max(population, 1.0) / _CORE_GROWTH_FROM) / np.log(
        _CORE_GROWTH_TO / _CORE_GROWTH_FROM
    )
    return _CORE_FRACTION + (_CORE_FRACTION_MAX - _CORE_FRACTION) * float(np.clip(growth, 0, 1))


def _footprint_km(population: np.ndarray | float) -> np.ndarray:
    """Footprint radius of places of ``population`` people (see the module docstring)."""
    pop = np.maximum(np.asarray(population, dtype=np.float64), 0.0)
    return np.clip(_FOOTPRINT_K * np.sqrt(pop / 1000.0), _FOOTPRINT_MIN_KM, _FOOTPRINT_MAX_KM)


def _to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _chord(km: float) -> float:
    """Straight-line distance on the unit sphere for a great-circle distance of ``km``."""
    return float(2.0 * np.sin(min(km / _EARTH_KM, np.pi) / 2.0))


def _chord_to_km(chord: np.ndarray) -> np.ndarray:
    return 2.0 * _EARTH_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2.0 * _EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class _Table:
    """Columnar copy of a GeoNames city list plus lookup indexes (raw dicts are dropped)."""

    geoid: list[int]
    names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    pop: np.ndarray
    cc: list[str]
    adm1: list[str]
    primary: dict[str, list[int]] = field(default_factory=dict)
    alt: dict[str, list[int]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class _Reading:
    """One ranked reading of a name."""

    rank: int
    pop: int
    tiebreak: tuple[str, ...]
    place: PlaceFilter


def _geonames_file(min_population: int) -> int:
    """Largest bundled city list whose threshold does not exceed ``min_population``."""
    fitting = [f for f in _GEONAMES_FILES if f <= min_population]
    return max(fitting) if fitting else _GEONAMES_FILES[0]


def _import_geonamescache() -> Any:
    try:
        import geonamescache
    except ImportError as exc:
        raise ModelUnavailableError(
            "place lookup needs the 'geonamescache' package: pip install 'facesorter[search-core]'"
        ) from exc
    return geonamescache


def _load_table(file_threshold: int, skip: Iterable[int] = (), with_alt: bool = True) -> _Table:
    """Read one ``geonamescache`` city list into a ``_Table`` (``skip``: geonameids to omit)."""
    geonamescache = _import_geonamescache()
    skipped = set(skip)
    cities = geonamescache.GeonamesCache(min_city_population=file_threshold).get_cities()
    geoid: list[int] = []
    names: list[str] = []
    lat: list[float] = []
    lon: list[float] = []
    pop: list[int] = []
    cc: list[str] = []
    adm1: list[str] = []
    primary: dict[str, list[int]] = {}
    alt: dict[str, list[int]] = {}
    for key, city in cities.items():
        gid = int(city.get("geonameid", key))
        if gid in skipped:
            continue
        i = len(names)
        geoid.append(gid)
        names.append(city["name"])
        lat.append(float(city["latitude"]))
        lon.append(float(city["longitude"]))
        population = int(city.get("population") or 0)
        pop.append(population)
        cc.append(city["countrycode"])
        adm1.append(city.get("admin1code") or "")
        primary.setdefault(_norm(city["name"]), []).append(i)
        if with_alt and population >= _ALT_MIN_POPULATION:
            for a in city.get("alternatenames", ()):
                if a.isascii():
                    key = a.lower().translate(_PUNCT)
                    if " " in key:  # same normalisation as the primary names
                        key = _norm(key)
                    if len(key) >= _ALT_MIN_LENGTH:
                        alt.setdefault(key, []).append(i)
    return _Table(
        geoid=geoid,
        names=names,
        lat=np.array(lat, dtype=np.float64),
        lon=np.array(lon, dtype=np.float64),
        pop=np.array(pop, dtype=np.int64),
        cc=cc,
        adm1=adm1,
        primary=primary,
        alt=alt,
    )


class OfflineGazetteer:
    """Offline ``Gazetteer`` over GeoNames cities. See the module docstring for the rules.

    Args:
        min_population: smallest population a place needs to be labelable (default 15 000).
        max_city_km: a coordinate farther than this from every plausible city gets no city.
        max_country_km: farther than this from every labelable place gives ``None``; between
            ``max_city_km`` and here, a country (and region) label when unambiguous.
    """

    def __init__(
        self,
        min_population: int = DEFAULT_MIN_POPULATION,
        max_city_km: float = 50.0,
        max_country_km: float = 250.0,
    ) -> None:
        if min_population <= 0:
            raise ValueError("min_population must be positive")
        if not 0 < max_city_km <= max_country_km:
            raise ValueError("need 0 < max_city_km <= max_country_km")
        self.min_population = int(min_population)
        self.max_city_km = float(max_city_km)
        self.max_country_km = float(max_country_km)

        gc = _import_geonamescache().GeonamesCache()
        self._countries: dict[str, str] = {
            cc: str(c["name"]).strip() for cc, c in gc.get_countries().items()
        }
        self._country_iso3 = {
            str(c.get("iso3", "")).upper(): cc for cc, c in gc.get_countries().items()
        }
        self._regions: dict[tuple[str, str], str] = {}
        self._abbreviations: dict[str, list[tuple[str, str]]] = {
            k: list(v) for k, v in _REGION_ABBREVIATIONS.items()
        }
        for code, state in gc.get_us_states().items():
            self._regions[("US", code)] = str(state["name"])
            self._abbreviations.setdefault(code, []).append(("US", code))
        for cc, table in _STATIC_REGIONS.items():
            for code, region_name in table.items():
                self._regions[(cc, code)] = region_name
        self._region_by_name: dict[str, list[tuple[str, str]]] = {}
        for (cc, code), region_name in sorted(self._regions.items()):
            self._region_by_name.setdefault(_norm(region_name), []).append((cc, code))
        self._country_by_name: dict[str, str] = {}
        for cc, country_name in self._countries.items():
            self._country_by_name.setdefault(_norm(country_name), cc)
        for alias, cc in _COUNTRY_ALIASES.items():
            self._country_by_name.setdefault(alias, cc)
        for alias, region in _REGION_ALIAS_INDEX.items():
            if region in self._regions:
                self._region_by_name.setdefault(alias, []).append(region)
        self._notable: dict[str, list[tuple[tuple[str, ...], PlaceFilter]]] = {}
        for names, ccs, nlat, nlon, radius in _NOTABLE_PLACES:
            for notable_name in names:
                shell = PlaceFilter(near=(nlat, nlon, radius), text=names[0])
                self._notable.setdefault(_norm(notable_name), []).append((ccs, shell))

        file_threshold = _geonames_file(self.min_population)
        self._core = _load_table(file_threshold)
        self._file_threshold = file_threshold
        self._wide: _Table | None = self._core if file_threshold <= _WIDE_FILE else None
        self._build_labelable()
        log.debug(
            "gazetteer: %d places, %d labelable (min_population=%d)",
            len(self._core),
            len(self._lab_rows),
            self.min_population,
        )

    # ------------------------------------------------------------------ labelling

    def _build_labelable(self) -> None:
        """Absorb neighbourhoods into bigger places, then index what is left."""
        t = self._core
        big = np.flatnonzero(t.pop >= self.min_population)
        xyz = _to_xyz(t.lat[big], t.lon[big])
        tree = _make_tree(xyz)
        pop = t.pop[big]
        absorbed = np.array([_is_composite(t.names[r]) for r in big], dtype=bool)
        for c in np.argsort(-pop, kind="stable"):
            if absorbed[c] or pop[c] < _CORE_RATIO * self.min_population:
                continue
            reach = float(_footprint_km(pop[c]))
            near = np.asarray(tree.query_ball_point(xyz[c], _chord(reach)), dtype=np.int64)
            near = near[(near != c) & ~absorbed[near]]
            if near.size == 0:
                continue
            rows = big[near]
            home = big[c]
            same_region = np.array(
                [
                    t.cc[r] == t.cc[home] and (t.adm1[r] == t.adm1[home] or t.cc[r] in _CITY_STATES)
                    for r in rows
                ],
                dtype=bool,
            )
            dist = _haversine_km(t.lat[home], t.lon[home], t.lat[rows], t.lon[rows])
            ratio = pop[c] / np.maximum(pop[near], 1)
            outskirts = (ratio >= _OUTSKIRTS_RATIO) & (dist <= reach)
            core = (ratio >= _CORE_RATIO) & (dist <= _core_fraction(pop[c]) * reach)
            absorbed[near[same_region & (outskirts | core)]] = True
        keep = np.flatnonzero(~absorbed)
        self._lab_rows = big[keep]  # row indexes into self._core
        self._lab_is = np.zeros(len(t), dtype=bool)
        self._lab_is[self._lab_rows] = True
        self._lab_footprint = _footprint_km(t.pop[self._lab_rows])
        self._lab_cc = np.array([t.cc[r] for r in self._lab_rows])
        self._lab_adm1 = np.array([t.adm1[r] for r in self._lab_rows])
        self._lab_labels = [self._row_label(t, int(r)) for r in self._lab_rows]
        self._tree: cKDTree | None = (
            _make_tree(_to_xyz(t.lat[self._lab_rows], t.lon[self._lab_rows]))
            if len(self._lab_rows)
            else None
        )

    def _country_name(self, cc: str) -> str | None:
        return self._countries.get(cc)

    def _region_name(self, cc: str, code: str) -> str | None:
        return self._regions.get((cc, code))

    def _row_label(self, table: _Table, i: int) -> PlaceLabel:
        cc = table.cc[i]
        return PlaceLabel(
            city=table.names[i],
            region=self._region_name(cc, table.adm1[i]),
            country=self._country_name(cc),
            country_code=cc,
        )

    def label_for(self, lat: float, lon: float) -> PlaceLabel | None:
        """Label one coordinate, or ``None`` if nothing plausible is near."""
        return self.label_many([lat], [lon])[0]

    def label_many(self, lats: Sequence[float], lons: Sequence[float]) -> list[PlaceLabel | None]:
        """Label many coordinates at once (vectorised). Invalid coordinates give ``None``."""
        lat = np.asarray(lats, dtype=np.float64).ravel()
        lon = np.asarray(lons, dtype=np.float64).ravel()
        if lat.shape != lon.shape:
            raise ValueError("lats and lons must have the same length")
        out: list[PlaceLabel | None] = [None] * lat.size
        valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        todo = np.flatnonzero(valid)
        if todo.size == 0 or self._tree is None:
            return out
        points = _to_xyz(lat[todo], lon[todo])
        k = min(_NEAREST_K, len(self._lab_rows))
        dist, nn = self._tree.query(points, k=k, distance_upper_bound=_chord(self.max_country_km))
        dist = dist.reshape(todo.size, k)
        nn = nn.reshape(todo.size, k)
        found = np.isfinite(dist)
        nn = np.where(found, nn, 0)
        km = np.where(found, _chord_to_km(np.where(found, dist, 0.0)), np.inf)
        best, has_city = self._choose(nn, km, found)
        chosen = np.where(has_city, nn[np.arange(todo.size), best], -1)
        if k < len(self._lab_rows):
            # The k nearest places may all be small suburbs of a big city farther out (Yankee
            # Stadium: the Bronx's neighbours crowd out New York City), so rows whose k-th
            # neighbour is still within city range are redone with an exact radius query.
            for row in np.flatnonzero(found[:, -1] & (km[:, -1] <= self.max_city_km)):
                ball = np.asarray(
                    self._tree.query_ball_point(points[row], _chord(self.max_city_km)),
                    dtype=np.int64,
                ).reshape(1, -1)
                ball_km = _chord_to_km(np.linalg.norm(self._tree.data[ball] - points[row], axis=-1))
                pick, ok = self._choose(ball, ball_km, np.ones(ball.shape, dtype=bool))
                chosen[row] = ball[0, pick[0]] if ok[0] else -1
        for row in range(todo.size):
            if chosen[row] >= 0:
                out[todo[row]] = self._lab_labels[int(chosen[row])]
            elif found[row].any():
                out[todo[row]] = self._country_label(nn[row][found[row]], km[row][found[row]])
        return out

    def _choose(
        self, nn: np.ndarray, km: np.ndarray, found: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pick, per row, the labelable place a coordinate is "in" (see the module docstring).

        Places whose footprint contains the point beat those that do not (nearest first, so
        Windsor's centre is Windsor, not Detroit 2 km away); otherwise the smallest
        ``distance / footprint`` wins.
        """
        footprint = self._lab_footprint[nn]
        reach = np.minimum(np.maximum(_REACH_MIN_KM, _REACH_FACTOR * footprint), self.max_city_km)
        eligible = found & (km <= reach)
        key = np.where(km <= footprint, km, _OUTSIDE_FOOTPRINT + km / footprint)
        key = np.where(eligible, key, np.inf)
        best = np.argmin(key, axis=1)
        return best, np.isfinite(key[np.arange(len(best)), best])

    def _country_label(self, neighbours: np.ndarray, km: np.ndarray) -> PlaceLabel | None:
        """Country (and region) label when one country is clearly the closest.

        ``neighbours`` are labelable places within range, nearest first, ``km`` their
        distances. The nearest place's country wins if every place of another country is
        at least ``1 / _COUNTRY_MARGIN`` times as far;
        the region is kept only if the same-country places within ``_REGION_MARGIN`` times
        the nearest distance agree on it (Zermatt: Sierre 34 km vs
        Domodossola 43 km -> Switzerland); otherwise the point is too close to a border.
        """
        ccs = self._lab_cc[neighbours]
        cc = str(ccs[0])
        foreign = km[ccs != cc]
        if foreign.size and km[0] > _COUNTRY_MARGIN * foreign.min():
            return None
        same = neighbours[(ccs == cc) & (km <= _REGION_MARGIN * km[0])]
        regions = {self._region_name(cc, a) for a in self._lab_adm1[same].tolist()}
        region = regions.pop() if len(regions) == 1 else None
        return PlaceLabel(city=None, region=region, country=self._country_name(cc), country_code=cc)

    # ------------------------------------------------------------------ name resolution

    def resolve(self, name: str) -> PlaceFilter | None:
        """Best reading of a place name from a query, or ``None`` if not understood."""
        found = self.candidates(name)
        return found[0] if found else None

    def candidates(self, name: str) -> list[PlaceFilter]:
        """All readings of ``name``, best first (see the module docstring for the order)."""
        text = " ".join(str(name).split())
        if not text or len(text) > _MAX_NAME_LENGTH:
            return []
        hits = self._hits(text)
        hits.sort(key=lambda h: (h.rank, -h.pop, h.tiebreak))
        unique: list[PlaceFilter] = []
        for hit in hits:
            if hit.place not in unique:
                unique.append(hit.place)
            if len(unique) >= _MAX_CANDIDATES:
                break
        return unique

    def _hits(self, text: str) -> list[_Reading]:
        whole = _norm(text.replace(",", " "))
        hits = self._alias_hits(whole, text)
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts or len(parts) > _MAX_PARTS:
            return hits
        # "Paris, Texas" is name + qualifier; "Misato, Saitama" is one GeoNames name.
        for split in range(1, len(parts) + 1):
            hits += self._named_hits(", ".join(parts[:split]), parts[split:], text)
        if not hits and len(parts) == 1:
            hits += self._trailing_qualifier_hits(parts[0], text)
        return hits

    def _trailing_qualifier_hits(self, part: str, text: str) -> list[_Reading]:
        """ "Key West Florida": retry with a known region/country as the last 1-3 words."""
        words = part.split()
        for take in (1, 2, 3):
            if len(words) <= take:
                break
            head, tail = " ".join(words[:-take]), " ".join(words[-take:])
            if self._scope(tail, qualifier=True) is None:
                continue
            hits = self._named_hits(head, [tail], text)
            if hits:
                return hits
        return []

    def _alias_hits(self, whole: str, text: str) -> list[_Reading]:
        alias = _CITY_ALIASES.get(whole)
        if alias is None:
            return []
        city, cc = alias
        rows = [i for i in self._core.primary.get(_norm(city), []) if self._core.cc[i] == cc]
        if not rows:
            return []
        best = max(rows, key=lambda i: self._core.pop[i])
        return [self._city_hit(self._core, best, text, _RANK_ALIAS)]

    def _scope(self, raw: str, qualifier: bool) -> tuple[set[str], set[tuple[str, str]]] | None:
        """Countries and regions a word could denote, or ``None`` if it is neither."""
        key = _norm(raw)
        countries: set[str] = set()
        regions: set[tuple[str, str]] = set()
        cc = self._country_by_name.get(key)
        if cc is not None:
            countries.add(cc)
        token = raw.strip().replace(".", "")
        if token.isalpha() and (qualifier or token.isupper()):
            upper = token.upper()
            if len(token) == 2 and upper in self._countries:
                countries.add(upper)
            elif len(token) == 3 and upper in self._country_iso3:
                countries.add(self._country_iso3[upper])
            # A bare capitalised code ("CA") is a country first; regions only as qualifiers.
            if len(token) <= 3 and (qualifier or not countries):
                regions.update(self._abbreviations.get(upper, ()))
        regions.update(self._region_by_name.get(key, ()))
        if not countries and not regions:
            return None
        return countries, regions

    def _named_hits(self, first: str, qualifiers: list[str], text: str) -> list[_Reading]:
        scopes = []
        for q in qualifiers:
            scope = self._scope(q, qualifier=True)
            if scope is None:
                return []  # an unknown qualifier: never guess
            scopes.append(scope)
        key = _norm(first)
        keys = [key]
        if key.startswith("the "):
            keys.append(key[4:])
        hits: list[_Reading] = self._notable_hits(keys, scopes, text)
        if not scopes:
            hits += self._country_hits(first, key, text)
        hits += self._region_hits(first, key, scopes, text)
        hits += self._city_hits(keys, scopes, text)
        return hits

    def _notable_hits(
        self,
        keys: list[str],
        scopes: list[tuple[set[str], set[tuple[str, str]]]],
        text: str,
    ) -> list[_Reading]:
        out = []
        for key in keys:
            for ccs, shell in self._notable.get(key, ()):
                nearby = self._regions_near(shell.near) if scopes else set()
                if all(
                    any(cc in countries for cc in ccs) or bool(regions & nearby)
                    for countries, regions in scopes
                ):
                    place = PlaceFilter(near=shell.near, text=text)
                    out.append(_Reading(_RANK_ALIAS, 0, ccs, place))
        return out

    def _regions_near(self, near: tuple[float, float, float]) -> set[tuple[str, str]]:
        """Regions of the labelable places around a curated destination (``(cc, admin1)``).

        Lets "Lake Tahoe, California" match while "Mount Rainier, Maryland" does not.
        """
        if self._tree is None:
            return set()
        lat, lon, radius = near
        point = _to_xyz(np.array([lat]), np.array([lon]))[0]
        found = self._tree.query_ball_point(point, _chord(radius + _NOTABLE_REGION_MARGIN_KM))
        return {(self._lab_cc[i], self._lab_adm1[i]) for i in found}

    def _country_hits(self, raw: str, key: str, text: str) -> list[_Reading]:
        scope = self._scope(raw, qualifier=False)
        if scope is None:
            return []
        out = []
        for cc in sorted(scope[0]):
            country = self._country_name(cc)
            if country:
                out.append(
                    _Reading(_RANK_COUNTRY, 0, (cc,), PlaceFilter(country=country, text=text))
                )
        return out

    def _region_hits(
        self,
        raw: str,
        key: str,
        scopes: list[tuple[set[str], set[tuple[str, str]]]],
        text: str,
    ) -> list[_Reading]:
        found = list(self._region_by_name.get(key, ()))
        token = raw.strip().replace(".", "")
        if 2 <= len(token) <= 3 and token.isalpha() and token.isupper() and not scopes:
            scope = self._scope(raw, qualifier=False)
            if scope:
                found += sorted(scope[1])
        out = []
        for cc, code in dict.fromkeys(found):
            if any(cc not in countries for countries, _ in scopes):
                continue
            region, country = self._regions[(cc, code)], self._country_name(cc)
            out.append(
                _Reading(
                    _RANK_REGION,
                    0,
                    (cc, code),
                    PlaceFilter(region=region, country=country, text=text),
                )
            )
        return out

    def _city_hits(
        self,
        keys: list[str],
        scopes: list[tuple[set[str], set[tuple[str, str]]]],
        text: str,
    ) -> list[_Reading]:
        hits = self._table_hits(self._core, keys, scopes, text)
        if not hits:
            hits = self._table_hits(self._wide_table(), keys, scopes, text, primary_only=True)
        return hits

    def _table_hits(
        self,
        table: _Table,
        keys: list[str],
        scopes: list[tuple[set[str], set[tuple[str, str]]]],
        text: str,
        primary_only: bool = False,
    ) -> list[_Reading]:
        rows: dict[int, int] = {}  # row -> rank (primary-name matches beat alternate names)
        for key in keys:
            primary = table.primary.get(key, ())
            for i in primary:
                rows[i] = _RANK_CITY
            if not primary_only:
                biggest = max((int(table.pop[i]) for i in primary), default=0)
                for i in table.alt.get(key, ()):
                    # An alternate name of a far bigger city is its nickname or exonym
                    # (Calcutta -> Kolkata beats the hamlet that is really named Calcutta).
                    famous = table.pop[i] >= _NICKNAME_RATIO * max(biggest, 1)
                    rows.setdefault(i, _RANK_CITY if famous else _RANK_CITY_ALT)
        hits = []
        for i, rank in rows.items():
            cc, adm1 = table.cc[i], table.adm1[i]
            if all(cc in countries or (cc, adm1) in regions for countries, regions in scopes):
                hits.append(self._city_hit(table, i, text, rank))
        return hits

    def _wide_table(self) -> _Table:
        """Places of 5 000+ people, loaded on first need (adds those not in the main list)."""
        if self._wide is None:
            known = self._core.geoid
            self._wide = _load_table(_WIDE_FILE, skip=known, with_alt=False)
            log.debug("gazetteer: loaded %d extra small places", len(self._wide))
        return self._wide

    def _city_hit(self, table: _Table, i: int, text: str, rank: int) -> _Reading:
        label = self._row_label(table, i)
        labelable = table is self._core and bool(self._lab_is[i])
        if labelable:
            place = PlaceFilter(
                city=label.city, region=label.region, country=label.country, text=text
            )
        else:
            radius = max(_NEAR_MIN_KM, float(_footprint_km(table.pop[i])))
            place = PlaceFilter(
                near=(round(float(table.lat[i]), 5), round(float(table.lon[i]), 5), radius),
                text=text,
            )
        tiebreak = (table.cc[i], table.names[i], table.adm1[i])
        return _Reading(rank, int(table.pop[i]), tiebreak, place)


def _make_tree(xyz: np.ndarray) -> cKDTree:
    from scipy.spatial import cKDTree

    return cKDTree(xyz)


@lru_cache(maxsize=4)
def build_gazetteer(min_population: int = DEFAULT_MIN_POPULATION) -> OfflineGazetteer:
    """Return a shared ``OfflineGazetteer`` (built once per ``min_population``)."""
    return OfflineGazetteer(min_population=min_population)
