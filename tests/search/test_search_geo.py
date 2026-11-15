"""
tests/search/test_search_geo.py
Offline gazetteer: coordinate labelling accuracy on a hand-made world set, name resolution,
and agreement between the two. Uses the installed ``geonamescache`` data only (no network).

The accuracy tests depend on that data (pinned only as ``geonamescache>=3.0``): a release can
change populations, names or admin1 codes. The landmark thresholds are therefore tolerant
(a few misses are allowed), while the region-code tables are checked exactly because a wrong
code would silently mislabel photos. Nothing here uses wall-clock assertions.
"""

from __future__ import annotations

import math
import sys

import pytest

pytest.importorskip("geonamescache")
pytest.importorskip("scipy")

from facesorter.search import geo  # noqa: E402
from facesorter.search.errors import ModelUnavailableError  # noqa: E402
from facesorter.search.geo import OfflineGazetteer, build_gazetteer  # noqa: E402
from facesorter.search.types import Gazetteer, PlaceFilter  # noqa: E402

# (name, lat, lon, expected). expected: a city name, "country:<Name>", "region:<Name>,<Country>"
# or None (no label). Coordinates are well-known public landmarks, not personal photos.
LANDMARKS = [
    ("Eiffel Tower", 48.8584, 2.2945, "Paris"),
    ("Louvre", 48.8606, 2.3376, "Paris"),
    ("Montmartre", 48.8867, 2.3431, "Paris"),
    ("Golden Gate Bridge", 37.8199, -122.4783, "San Francisco"),
    ("Fisherman's Wharf", 37.8080, -122.4177, "San Francisco"),
    ("Times Square", 40.7580, -73.9855, "New York City"),
    ("Brooklyn Museum", 40.6712, -73.9636, "New York City"),
    ("Statue of Liberty", 40.6892, -74.0445, "New York City"),
    ("Shibuya Crossing", 35.6595, 139.7005, "Tokyo"),
    ("Tokyo Tower", 35.6586, 139.7454, "Tokyo"),
    ("Big Ben", 51.5007, -0.1246, "London"),
    ("Colosseum", 41.8902, 12.4922, "Rome"),
    ("Sagrada Familia", 41.4036, 2.1744, "Barcelona"),
    ("Brandenburg Gate", 52.5163, 13.3777, "Berlin"),
    ("Sydney Opera House", -33.8568, 151.2153, "Sydney"),
    ("Christ the Redeemer", -22.9519, -43.2105, "Rio de Janeiro"),
    ("Taj Mahal", 27.1751, 78.0421, "Agra"),
    ("Burj Khalifa", 25.1972, 55.2744, "Dubai"),
    ("Gateway of India", 18.9220, 72.8347, "Mumbai"),
    ("Mehrangarh Fort", 26.2979, 73.0184, "Jodhpur"),
    ("Vidhana Soudha", 12.9796, 77.5908, "Bengaluru"),
    ("Space Needle", 47.6205, -122.3493, "Seattle"),
    ("Liberty Bell", 39.9496, -75.1503, "Philadelphia"),
    ("Duval Street", 24.5551, -81.7800, "Key West"),
    ("Amherst College", 42.3709, -72.5170, "Amherst"),
    ("Sunnyvale downtown", 37.3688, -122.0363, "Sunnyvale"),
    ("Milpitas", 37.4323, -121.8996, "Milpitas"),
    ("Mountain View, Castro St", 37.3894, -122.0819, "Mountain View"),
    ("Santa Clara Mission", 37.3541, -121.9552, "Santa Clara"),
    ("San Jose downtown", 37.3382, -121.8863, "San Jose"),
    ("Palo Alto, University Ave", 37.4419, -122.1430, "Palo Alto"),
    ("Santa Monica Pier", 34.0094, -118.4973, "Santa Monica"),
    ("Hollywood Sign", 34.1341, -118.3215, "Los Angeles"),
    ("Hoboken waterfront", 40.7440, -74.0324, "Hoboken"),
    ("Zurich Bahnhofstrasse", 47.3769, 8.5417, "Zürich"),
    ("Reykjavik Hallgrimskirkja", 64.1417, -21.9266, "Reykjavík"),
    ("Table Mountain", -33.9628, 18.4098, "Cape Town"),
    ("Marina Bay", 1.2834, 103.8607, "Singapore"),
    ("Grand Palace Bangkok", 13.7500, 100.4913, "Bangkok"),
    ("Hagia Sophia", 41.0086, 28.9802, "Istanbul"),
    ("Red Square", 55.7539, 37.6208, "Moscow"),
    ("Zocalo", 19.4326, -99.1332, "Mexico City"),
    ("CN Tower", 43.6426, -79.3871, "Toronto"),
    ("Stanley Park", 49.3017, -123.1417, "Vancouver"),
    ("Lincoln Memorial", 38.8893, -77.0502, "Washington"),
    ("Texas State Capitol", 30.2747, -97.7404, "Austin"),
    ("Fenway Park", 42.3467, -71.0972, "Boston"),
    ("Berkeley campus", 37.8719, -122.2585, "Berkeley"),
    ("Santa Cruz Boardwalk", 36.9643, -122.0178, "Santa Cruz"),
    ("Leaning Tower of Pisa", 43.7230, 10.3966, "Pisa"),
    ("India Gate", 28.6129, 77.2295, "Delhi"),
    ("Fushimi Inari", 34.9671, 135.7727, "Kyoto"),
    ("Radcliffe Camera", 51.7536, -1.2540, "Oxford"),
    ("St Mark's Square", 45.4343, 12.3388, "Venice"),
    # Dense metros, where the nearest few places are all neighbourhoods or suburbs.
    ("Yankee Stadium", 40.8296, -73.9262, "New York City"),
    ("Bronx Zoo", 40.8506, -73.8769, "New York City"),
    ("Staten Island Ferry", 40.5795, -74.1502, "New York City"),
    ("JFK Airport", 40.6413, -73.7781, "New York City"),
    ("Flushing Meadows", 40.7466, -73.8448, "New York City"),
    ("Brooklyn Heights", 40.6960, -73.9936, "New York City"),
    ("Tsim Sha Tsui, Kowloon", 22.2988, 114.1722, "Hong Kong"),
    ("Shinjuku", 35.6896, 139.7006, "Tokyo"),
    ("Lake Merritt", 37.8024, -122.2563, "Oakland"),
    ("Windsor (GeoNames centre)", 42.3001, -83.0165, "Windsor"),
    ("Johor Bahru (GeoNames centre)", 1.4655, 103.7578, "Johor Bahru"),
    ("Sausalito", 37.8591, -122.4853, "San Francisco"),  # nearest labelable city
    ("Waikiki", 21.2766, -157.8275, "Honolulu"),
    ("Baga Beach, Goa", 15.5524, 73.7517, "Calangute"),
    # Remote places: a country (and region) label is the honest answer.
    ("Zermatt", 46.0207, 7.7491, "country:Switzerland"),
    ("Machu Picchu", -13.1631, -72.5450, "country:Peru"),
    ("Grand Canyon South Rim", 36.0544, -112.1401, "region:Arizona,United States"),
    ("Whistler village", 50.1163, -122.9574, "region:British Columbia,Canada"),
    ("Iceland highlands", 64.8, -18.5, "country:Iceland"),
    # Nothing plausible nearby.
    ("Mid-Atlantic", 0.0, -30.0, None),
    ("South Pacific", -30.0, -130.0, None),
    ("Southern Ocean", -60.0, 20.0, None),
    ("North Pole", 89.9, 0.0, None),
]

# Cases where a different answer is also reasonable (a real neighbouring city or the
# parent metropolis); a miss here counts as acceptable, not as an error.
ACCEPTABLE = {
    "Amherst College": {"Amherst"},
    "Hoboken waterfront": {"Hoboken", "New York City", "Jersey City"},
    "Santa Monica Pier": {"Santa Monica", "Los Angeles"},
    "Montmartre": {"Paris"},
}

MIN_EXACT_ACCURACY = 0.95
MIN_ACCEPTABLE_ACCURACY = 0.98


def _matches(label, expected):
    if expected is None:
        return label is None
    if label is None:
        return False
    if expected.startswith("country:"):
        return label.city is None and label.country == expected[len("country:") :]
    if expected.startswith("region:"):
        region, country = expected[len("region:") :].split(",")
        return label.city is None and label.region == region and label.country == country
    return label.city == expected


@pytest.fixture(scope="module")
def gaz():
    return build_gazetteer()


def test_satisfies_protocol_and_build_is_cached(gaz):
    assert isinstance(gaz, Gazetteer)
    assert build_gazetteer() is build_gazetteer() is gaz
    assert build_gazetteer(20_000) is not gaz


def test_missing_geonamescache_raises_typed_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "geonamescache", None)  # makes ``import`` fail
    with pytest.raises(ModelUnavailableError):
        OfflineGazetteer()
    with pytest.raises(ModelUnavailableError):
        geo._load_table(15000)


def test_label_accuracy_on_world_landmarks(gaz):
    labels = gaz.label_many([p[1] for p in LANDMARKS], [p[2] for p in LANDMARKS])
    exact, acceptable, misses = 0, 0, []
    for (name, _lat, _lon, expected), label in zip(LANDMARKS, labels):
        if _matches(label, expected):
            exact += 1
            acceptable += 1
        elif label is not None and label.city in ACCEPTABLE.get(name, ()):
            acceptable += 1
        else:
            misses.append((name, expected, label and label.display))
    total = len(LANDMARKS)
    assert total >= 70
    assert exact / total >= MIN_EXACT_ACCURACY, misses
    assert acceptable / total >= MIN_ACCEPTABLE_ACCURACY, misses


def test_label_for_matches_label_many(gaz):
    lats = [p[1] for p in LANDMARKS]
    lons = [p[2] for p in LANDMARKS]
    assert [gaz.label_for(a, b) for a, b in zip(lats, lons)] == gaz.label_many(lats, lons)


def test_label_fields(gaz):
    label = gaz.label_for(37.7749, -122.4194)
    assert (label.city, label.region, label.country, label.country_code) == (
        "San Francisco",
        "California",
        "United States",
        "US",
    )
    assert label.display == "San Francisco, California, United States"
    tokyo = gaz.label_for(35.6762, 139.6503)
    assert (tokyo.city, tokyo.region, tokyo.country) == ("Tokyo", None, "Japan")
    assert gaz.label_for(43.6532, -79.3832).region == "Ontario"
    assert gaz.label_for(-37.8136, 144.9631).region == "Victoria"
    assert gaz.label_for(55.9533, -3.1883).region == "Scotland"


def test_neighbouring_cities_stay_distinct(gaz):
    # ~10-15 km apart in the south bay; none may collapse into San Jose.
    centres = {
        "Sunnyvale": (37.3688, -122.0363),
        "Milpitas": (37.4323, -121.8996),
        "Mountain View": (37.3894, -122.0819),
        "Santa Clara": (37.3541, -121.9552),
        "San Jose": (37.3382, -121.8863),
    }
    for city, (lat, lon) in centres.items():
        assert gaz.label_for(lat, lon).city == city


def test_neighbourhoods_are_not_labels(gaz):
    labelled = {(lb.city, lb.region) for lb in gaz._lab_labels}
    for hood in ("Chinatown", "Parkside", "Paris 16 Passy", "Mission District"):
        assert not any(city == hood for city, _ in labelled)
    for borough in ("Manhattan", "Brooklyn", "Queens", "The Bronx"):
        assert (borough, "New York") not in labelled
    assert not any(" / " in (city or "") or "(" in (city or "") for city, _ in labelled)
    assert {
        ("San Francisco", "California"),
        ("Paris", "Île-de-France"),
        ("New York City", "New York"),
    } <= labelled
    assert ("Hoboken", "New Jersey") in labelled and ("Jersey City", "New Jersey") in labelled


def test_far_from_anything_is_none_not_a_distant_town(gaz):
    assert gaz.label_for(0.0, -30.0) is None  # 800+ km from land: no distant Brazilian town
    assert gaz.label_for(-51.0, -73.0) is None  # Patagonian icefield
    assert gaz.label_for(23.0, 10.0) is None  # central Sahara
    assert gaz.label_for(72.0, -40.0) is None  # Greenland ice sheet


def test_remote_land_gets_a_country_label(gaz):
    iceland = gaz.label_for(64.8, -18.5)  # highland interior, > 50 km from any town
    assert iceland.city is None and iceland.country == "Iceland"
    off_portugal = gaz.label_for(38.7, -11.0)  # at sea, 150 km out
    assert (off_portugal.city, off_portugal.country_code) == (None, "PT")


def test_close_to_a_border_gets_no_country_guess(gaz):
    # Matterhorn summit: Swiss and Italian towns at similar distances, so no confident country.
    assert gaz.label_for(45.98, 7.66) is None


def test_invalid_coordinates(gaz):
    nan = float("nan")
    assert gaz.label_for(nan, 10.0) is None
    assert gaz.label_for(95.0, 10.0) is None
    assert gaz.label_for(10.0, 200.0) is None
    assert gaz.label_for(math.inf, 0.0) is None
    assert gaz.label_many([], []) == []
    mixed = gaz.label_many([nan, 48.8584], [0.0, 2.2945])
    assert mixed[0] is None and mixed[1].city == "Paris"
    with pytest.raises(ValueError):
        gaz.label_many([1.0, 2.0], [1.0])


def test_label_many_handles_thousands_of_points(gaz):
    lats = [37.0 + (i % 100) * 0.01 for i in range(5000)]
    lons = [-122.0 + (i // 100) * 0.01 for i in range(5000)]
    assert len(gaz.label_many(lats, lons)) == 5000


def test_constructor_validation():
    with pytest.raises(ValueError):
        OfflineGazetteer(min_population=0)
    with pytest.raises(ValueError):
        OfflineGazetteer(max_city_km=300.0, max_country_km=100.0)


def test_nothing_labelable():
    empty = OfflineGazetteer(min_population=10**9)
    assert empty.label_for(48.8584, 2.2945) is None
    assert empty.label_many([1.0], [1.0]) == [None]
    assert empty.resolve("Cornwall, England") is None  # no labelable places to vouch for it
    assert empty.resolve("Cornwall").near is not None


def test_lower_min_population_labels_smaller_places():
    small = OfflineGazetteer(min_population=5000)
    assert len(small._lab_rows) > len(build_gazetteer()._lab_rows)
    zermatt = small.label_for(46.0207, 7.7491)
    assert zermatt is not None and zermatt.city == "Zermatt"
    # a small labelable place now resolves as a city filter, not a near filter
    # (curated destinations still win over the plain place, so use a town without one)
    assert small.resolve("Visp").city == "Visp"


def test_geonames_file_choice():
    assert geo._geonames_file(15000) == 15000
    assert geo._geonames_file(10000) == 5000
    assert geo._geonames_file(1500) == 1000
    assert geo._geonames_file(10) == 500


# (country, admin1 code, a city GeoNames files under it, region name we emit). Every code in
# every built-in table must appear here: a code that cannot be verified is left out.
REGION_ANCHORS = [
    ("CA", "01", "Calgary", "Alberta"),
    ("CA", "02", "Vancouver", "British Columbia"),
    ("CA", "03", "Winnipeg", "Manitoba"),
    ("CA", "04", "Moncton", "New Brunswick"),
    ("CA", "05", "St. John's", "Newfoundland and Labrador"),
    ("CA", "07", "Halifax", "Nova Scotia"),
    ("CA", "08", "Toronto", "Ontario"),
    ("CA", "09", "Charlottetown", "Prince Edward Island"),
    ("CA", "10", "Montréal", "Quebec"),
    ("CA", "11", "Saskatoon", "Saskatchewan"),
    ("CA", "12", "Whitehorse", "Yukon"),
    ("CA", "13", "Yellowknife", "Northwest Territories"),
    ("CA", "14", "Iqaluit", "Nunavut"),
    ("AU", "01", "Canberra", "Australian Capital Territory"),
    ("AU", "02", "Sydney", "New South Wales"),
    ("AU", "03", "Darwin", "Northern Territory"),
    ("AU", "04", "Brisbane", "Queensland"),
    ("AU", "05", "Adelaide", "South Australia"),
    ("AU", "06", "Hobart", "Tasmania"),
    ("AU", "07", "Melbourne", "Victoria"),
    ("AU", "08", "Perth", "Western Australia"),
    ("GB", "ENG", "Manchester", "England"),
    ("GB", "SCT", "Glasgow", "Scotland"),
    ("GB", "WLS", "Cardiff", "Wales"),
    ("GB", "NIR", "Belfast", "Northern Ireland"),
    ("IN", "01", "Port Blair", "Andaman and Nicobar Islands"),
    ("IN", "02", "Vijayawada", "Andhra Pradesh"),
    ("IN", "03", "Guwahati", "Assam"),
    ("IN", "05", "Chandigarh", "Chandigarh"),
    ("IN", "07", "Delhi", "Delhi"),
    ("IN", "09", "Ahmedabad", "Gujarat"),
    ("IN", "10", "Gurugram", "Haryana"),
    ("IN", "11", "Shimla", "Himachal Pradesh"),
    ("IN", "12", "Srinagar", "Jammu and Kashmir"),
    ("IN", "13", "Kochi", "Kerala"),
    ("IN", "16", "Mumbai", "Maharashtra"),
    ("IN", "17", "Imphal", "Manipur"),
    ("IN", "18", "Shillong", "Meghalaya"),
    ("IN", "19", "Bengaluru", "Karnataka"),
    ("IN", "20", "Kohima", "Nagaland"),
    ("IN", "21", "Bhubaneswar", "Odisha"),
    ("IN", "22", "Puducherry", "Puducherry"),
    ("IN", "23", "Amritsar", "Punjab"),
    ("IN", "24", "Jaipur", "Rajasthan"),
    ("IN", "25", "Chennai", "Tamil Nadu"),
    ("IN", "26", "Agartala", "Tripura"),
    ("IN", "28", "Kolkata", "West Bengal"),
    ("IN", "29", "Gangtok", "Sikkim"),
    ("IN", "30", "Itanagar", "Arunachal Pradesh"),
    ("IN", "31", "Aizawl", "Mizoram"),
    ("IN", "33", "Panjim", "Goa"),
    ("IN", "34", "Patna", "Bihar"),
    ("IN", "35", "Bhopal", "Madhya Pradesh"),
    ("IN", "36", "Lucknow", "Uttar Pradesh"),
    ("IN", "37", "Raipur", "Chhattisgarh"),
    ("IN", "38", "Ranchi", "Jharkhand"),
    ("IN", "39", "Dehradun", "Uttarakhand"),
    ("IN", "40", "Hyderabad", "Telangana"),
    ("IN", "41", "Leh", "Ladakh"),
    ("IN", "52", "Silvassa", "Dadra and Nagar Haveli and Daman and Diu"),
    ("DE", "01", "Stuttgart", "Baden-Württemberg"),
    ("DE", "02", "Munich", "Bavaria"),
    ("DE", "03", "Bremen", "Bremen"),
    ("DE", "04", "Hamburg", "Hamburg"),
    ("DE", "05", "Frankfurt am Main", "Hesse"),
    ("DE", "06", "Hannover", "Lower Saxony"),
    ("DE", "07", "Köln", "North Rhine-Westphalia"),
    ("DE", "08", "Mainz", "Rhineland-Palatinate"),
    ("DE", "09", "Saarbrücken", "Saarland"),
    ("DE", "10", "Kiel", "Schleswig-Holstein"),
    ("DE", "11", "Potsdam", "Brandenburg"),
    ("DE", "12", "Rostock", "Mecklenburg-Vorpommern"),
    ("DE", "13", "Dresden", "Saxony"),
    ("DE", "14", "Magdeburg", "Saxony-Anhalt"),
    ("DE", "15", "Erfurt", "Thuringia"),
    ("DE", "16", "Berlin", "Berlin"),
    ("IT", "01", "Pescara", "Abruzzo"),
    ("IT", "02", "Matera", "Basilicata"),
    ("IT", "03", "Reggio Calabria", "Calabria"),
    ("IT", "04", "Naples", "Campania"),
    ("IT", "05", "Bologna", "Emilia-Romagna"),
    ("IT", "06", "Trieste", "Friuli-Venezia Giulia"),
    ("IT", "07", "Rome", "Lazio"),
    ("IT", "08", "Genoa", "Liguria"),
    ("IT", "09", "Milan", "Lombardy"),
    ("IT", "10", "Ancona", "Marche"),
    ("IT", "11", "Campobasso", "Molise"),
    ("IT", "12", "Turin", "Piedmont"),
    ("IT", "13", "Bari", "Apulia"),
    ("IT", "14", "Cagliari", "Sardinia"),
    ("IT", "15", "Palermo", "Sicily"),
    ("IT", "16", "Florence", "Tuscany"),
    ("IT", "17", "Trento", "Trentino-Alto Adige"),
    ("IT", "18", "Perugia", "Umbria"),
    ("IT", "19", "Aosta", "Aosta Valley"),
    ("IT", "20", "Verona", "Veneto"),
    ("FR", "11", "Paris", "Île-de-France"),
    ("FR", "24", "Tours", "Centre-Val de Loire"),
    ("FR", "27", "Dijon", "Bourgogne-Franche-Comté"),
    ("FR", "28", "Le Havre", "Normandy"),
    ("FR", "32", "Lille", "Hauts-de-France"),
    ("FR", "44", "Strasbourg", "Grand Est"),
    ("FR", "52", "Nantes", "Pays de la Loire"),
    ("FR", "53", "Rennes", "Brittany"),
    ("FR", "75", "Bordeaux", "Nouvelle-Aquitaine"),
    ("FR", "76", "Toulouse", "Occitanie"),
    ("FR", "84", "Lyon", "Auvergne-Rhône-Alpes"),
    ("FR", "93", "Marseille", "Provence-Alpes-Côte d'Azur"),
    ("FR", "94", "Ajaccio", "Corsica"),
    ("ES", "07", "Palma", "Balearic Islands"),
    ("ES", "27", "Logroño", "La Rioja"),
    ("ES", "29", "Madrid", "Community of Madrid"),
    ("ES", "31", "Murcia", "Murcia"),
    ("ES", "32", "Pamplona", "Navarre"),
    ("ES", "34", "Gijón", "Asturias"),
    ("ES", "39", "Santander", "Cantabria"),
    ("ES", "51", "Sevilla", "Andalusia"),
    ("ES", "52", "Zaragoza", "Aragon"),
    ("ES", "53", "Las Palmas de Gran Canaria", "Canary Islands"),
    ("ES", "54", "Albacete", "Castilla-La Mancha"),
    ("ES", "55", "Valladolid", "Castile and León"),
    ("ES", "56", "Barcelona", "Catalonia"),
    ("ES", "57", "Badajoz", "Extremadura"),
    ("ES", "58", "Vigo", "Galicia"),
    ("ES", "59", "Bilbao", "Basque Country"),
    ("ES", "60", "Valencia", "Valencian Community"),
    ("AT", "02", "Klagenfurt am Wörthersee", "Carinthia"),
    ("AT", "03", "Wiener Neustadt", "Lower Austria"),
    ("AT", "04", "Linz", "Upper Austria"),
    ("AT", "05", "Salzburg", "Salzburg"),
    ("AT", "06", "Graz", "Styria"),
    ("AT", "07", "Innsbruck", "Tyrol"),
    ("AT", "08", "Dornbirn", "Vorarlberg"),
    ("AT", "09", "Vienna", "Vienna"),
    ("US", "TX", "Austin", "Texas"),
    ("US", "DC", "Washington", "District of Columbia"),
]


@pytest.mark.parametrize(("cc", "code", "city", "region"), REGION_ANCHORS)
def test_region_code_matches_a_known_city(gaz, cc, code, city, region):
    found = []
    for table in (gaz._core, gaz._wide_table()):  # the second list adds towns under 15 000
        found += [
            (table.pop[i], table, i)
            for i in table.primary.get(geo._norm(city), [])
            if table.cc[i] == cc
        ]
    assert found, city
    _, table, best = max(found, key=lambda f: f[0])
    assert table.adm1[best] == code
    assert gaz._row_label(table, best).region == region


def test_every_built_in_region_code_has_an_anchor(gaz):
    anchored = {(cc, code) for cc, code, _, _ in REGION_ANCHORS}
    built_in = {(cc, code) for cc, table in geo._STATIC_REGIONS.items() for code in table}
    assert built_in == anchored - {("US", "TX"), ("US", "DC")}
    assert gaz.label_for(35.6762, 139.6503).region is None  # Japan: not verified, so no names


def test_labels_carry_region_names_outside_the_us(gaz):
    assert gaz.label_for(12.9716, 77.5946).region == "Karnataka"
    assert gaz.label_for(26.2979, 73.0184).region == "Rajasthan"
    assert gaz.label_for(48.1374, 11.5755).region == "Bavaria"
    assert gaz.label_for(41.9028, 12.4964).region == "Lazio"
    assert gaz.label_for(43.6532, -79.3832).region == "Ontario"
    assert gaz.label_for(-37.8136, 144.9631).region == "Victoria"
    assert gaz.label_for(55.9533, -3.1883).region == "Scotland"


def test_region_kept_when_close_neighbours_agree(gaz):
    # at sea off Cape Mendocino: the nearby towns are all in California
    assert gaz.label_for(40.25, -124.0).region == "California"


# ----------------------------------------------------------------------------- resolve


def _city(f: PlaceFilter):
    return (f.city, f.region, f.country)


def test_resolve_cities(gaz):
    assert _city(gaz.resolve("Paris")) == ("Paris", "Île-de-France", "France")
    assert _city(gaz.resolve("paris, texas")) == ("Paris", "Texas", "United States")
    assert _city(gaz.resolve("Paris, France")) == ("Paris", "Île-de-France", "France")
    assert _city(gaz.resolve("Austin, TX")) == ("Austin", "Texas", "United States")
    assert _city(gaz.resolve("Portland, Oregon")) == ("Portland", "Oregon", "United States")
    assert _city(gaz.resolve("Portland, Maine")) == ("Portland", "Maine", "United States")
    assert _city(gaz.resolve("Amherst, MA")) == ("Amherst", "Massachusetts", "United States")
    assert _city(gaz.resolve("Amherst, Massachusetts, USA")) == (
        "Amherst",
        "Massachusetts",
        "United States",
    )
    assert _city(gaz.resolve("Key West")) == ("Key West", "Florida", "United States")
    assert _city(gaz.resolve("Key West Florida")) == ("Key West", "Florida", "United States")
    assert _city(gaz.resolve("Paris France")) == ("Paris", "Île-de-France", "France")
    assert _city(gaz.resolve("Jodhpur")) == ("Jodhpur", "Rajasthan", "India")
    assert _city(gaz.resolve("the Hague")) == ("The Hague", None, "The Netherlands")
    assert _city(gaz.resolve("Mexico City")) == ("Mexico City", None, "Mexico")


def test_resolve_keeps_user_wording_as_text(gaz):
    assert gaz.resolve("  zurich ").text == "zurich"
    assert gaz.resolve("Paris,   Texas").text == "Paris, Texas"


def test_resolve_diacritics_case_and_punctuation(gaz):
    for query in ("Zurich", "ZÜRICH", "zürich", "Zürich"):
        assert gaz.resolve(query).city == "Zürich"
    assert gaz.resolve("Sao Paulo").city == "São Paulo"
    assert gaz.resolve("São Paulo").city == "São Paulo"
    assert gaz.resolve("St. Louis").city == gaz.resolve("Saint Louis").city
    assert gaz.resolve("Bangalore").city == "Bengaluru"
    assert gaz.resolve("Bengaluru").city == "Bengaluru"
    assert gaz.resolve("Bombay").city == "Mumbai"
    assert gaz.resolve("Munich").city == "Munich"
    assert gaz.resolve("Reykjavik").city == "Reykjavík"


def test_resolve_aliases(gaz):
    for alias, city in {
        "NYC": "New York City",
        "New York": "New York City",
        "new york city": "New York City",
        "SF": "San Francisco",
        "San Fran": "San Francisco",
        "LA": "Los Angeles",
        "L.A.": "Los Angeles",
        "Philly": "Philadelphia",
        "Vegas": "Las Vegas",
        "Washington DC": "Washington",
        "Washington, D.C.": "Washington",
        "DC": "Washington",
    }.items():
        resolved = gaz.resolve(alias)
        assert resolved.city == city, alias
        assert resolved.text == alias


def test_resolve_countries(gaz):
    for query, country in {
        "France": "France",
        "united states": "United States",
        "USA": "United States",
        "US": "United States",
        "U.S.A.": "United States",
        "America": "United States",
        "UK": "United Kingdom",
        "United Kingdom": "United Kingdom",
        "Great Britain": "United Kingdom",
        "Holland": "The Netherlands",
        "Czech Republic": "Czechia",
        "FR": "France",
        "DEU": "Germany",
        "Japan": "Japan",
        "south korea": "South Korea",
    }.items():
        resolved = gaz.resolve(query)
        assert (resolved.country, resolved.city, resolved.region) == (country, None, None), query


def test_resolve_regions(gaz):
    california = gaz.resolve("California")
    assert (california.region, california.country, california.city) == (
        "California",
        "United States",
        None,
    )
    assert gaz.resolve("california").region == "California"
    assert gaz.resolve("Ontario").region == "Ontario"  # not Ontario, CA the city
    assert gaz.resolve("British Columbia").region == "British Columbia"
    assert gaz.resolve("Scotland").country == "United Kingdom"
    assert gaz.resolve("New South Wales").country == "Australia"
    assert gaz.resolve("NY").region == "New York"
    assert gaz.resolve("Texas, USA").region == "Texas"
    assert gaz.resolve("CA").country == "Canada"  # a bare code is a country first


def test_resolve_ambiguity_rule_and_candidates(gaz):
    # country > region > city; larger cities first
    assert gaz.resolve("Georgia").country == "Georgia"
    assert gaz.resolve("Georgia").region is None
    assert [c.region for c in gaz.candidates("Georgia")][:2] == [None, "Georgia"]
    assert gaz.resolve("Washington").region == "Washington"
    washington = gaz.candidates("Washington")
    assert washington[1].city == "Washington" and washington[1].region == "District of Columbia"
    amherst = gaz.candidates("Amherst")
    assert [(c.city, c.region) for c in amherst[:2]] == [
        ("Amherst", "New York"),
        ("Amherst", "Massachusetts"),
    ]
    assert len(gaz.candidates("Springfield")) >= 3
    assert gaz.resolve("Springfield") == gaz.candidates("Springfield")[0]
    assert gaz.resolve("Springfield").region == "Missouri"
    # deterministic
    lower = [(c.city, c.region, c.country) for c in gaz.candidates("springfield")]
    assert [(c.city, c.region, c.country) for c in gaz.candidates("Springfield")] == lower


def test_candidates_are_unique_and_bounded(gaz):
    found = gaz.candidates("San Jose")
    assert len(found) == len(set(found)) <= 12
    assert found[0].city == "San Jose" and found[0].country == "United States"


def test_resolve_small_places_as_near_filters(gaz):
    whistler = gaz.resolve("Whistler")
    assert whistler.city is None and whistler.region is None and whistler.country is None
    lat, lon, radius = whistler.near
    assert abs(lat - 50.12) < 0.1 and abs(lon + 122.95) < 0.1
    assert 1 <= radius <= 30
    assert whistler.text == "Whistler"
    assert gaz.resolve("Zermatt").near is not None
    # neighbourhoods are not labels, so they resolve to a point
    brooklyn = gaz.resolve("Brooklyn")
    assert brooklyn.near is not None and abs(brooklyn.near[0] - 40.65) < 0.1
    assert gaz.resolve("Chinatown, California").near is not None
    # qualifiers apply to small places too
    assert gaz.resolve("Whistler, Canada").near is not None
    assert gaz.resolve("Whistler, France") is None


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "red car",
        "Tuesday",
        "sunset beach",
        "my dog",
        "dog",
        "car",
        "last summer",
        "Paris, Mars",
        "asdfghjkl",
        "Hokkaido",
        "Rajastan",
        "Tuscani",
        "a, b, c, d",
        "x" * 200,
        ",",
        "the",
    ],
)
def test_resolve_negatives(gaz, query):
    assert gaz.resolve(query) is None
    assert gaz.candidates(query) == []


def test_candidates_never_raise_on_odd_input(gaz):
    for odd in ("%%%", "((", "'", "-", "ñ", "日本", "Москва", "🙂"):
        gaz.candidates(odd)


# ------------------------------------------------------------------- label <-> resolve


def test_every_label_round_trips_through_resolve(gaz):
    table = gaz._core
    checked = 0
    for pos in range(0, len(gaz._lab_rows), 7):
        label = gaz._lab_labels[pos]
        qualifier = label.region or label.country
        found = gaz.candidates(f"{label.city}, {qualifier}")
        wanted = PlaceFilter(
            city=label.city,
            region=label.region,
            country=label.country,
            text=f"{label.city}, {qualifier}",
        )
        assert wanted in found, label
        checked += 1
    assert checked > 4000
    assert len(table) > 30000


def test_unqualified_label_round_trip_for_big_cities(gaz):
    # For the most populous labelable places the bare name resolves to that very city.
    order = sorted(range(len(gaz._lab_rows)), key=lambda p: -gaz._core.pop[gaz._lab_rows[p]])
    seen = set()
    for pos in order[:300]:
        label = gaz._lab_labels[pos]
        if label.city in seen:
            continue
        seen.add(label.city)
        resolved = gaz.resolve(label.city)
        assert resolved is not None
        assert resolved.city == label.city or resolved.city is None, label
        if resolved.city is not None:
            assert (resolved.region, resolved.country) == (label.region, label.country), label


def test_resolve_output_matches_label_spelling(gaz):
    pairs = [
        ((37.7749, -122.4194), "San Francisco"),
        ((48.8566, 2.3522), "Paris"),
        ((47.3769, 8.5417), "Zurich"),
        ((-23.55, -46.63), "Sao Paulo"),
        ((12.9716, 77.5946), "Bangalore"),
    ]
    for (lat, lon), query in pairs:
        label = gaz.label_for(lat, lon)
        resolved = gaz.resolve(query)
        assert (resolved.city, resolved.region, resolved.country) == (
            label.city,
            label.region,
            label.country,
        )


def test_alias_for_a_city_missing_from_the_data(gaz, monkeypatch):
    monkeypatch.setitem(geo._CITY_ALIASES, "qzx", ("Shangri-La Nowhere", "US"))
    assert gaz.resolve("qzx") is None


def test_wide_table_is_lazy_and_cached():
    fresh = OfflineGazetteer()
    assert fresh._wide is None
    fresh.resolve("Paris")
    assert fresh._wide is None  # a main-list hit never loads the small places
    fresh.resolve("Whistler")
    wide = fresh._wide
    assert wide is not None and fresh._wide_table() is wide
    assert not set(wide.geoid) & set(fresh._core.geoid)


# ------------------------------------------------------------- reviewer-driven regressions


def test_dense_metro_outer_boroughs_are_not_stolen_by_neighbours(gaz, monkeypatch):
    points = [(40.8296, -73.9262), (40.8506, -73.8769), (40.5795, -74.1502), (40.6413, -73.7781)]
    for lat, lon in points:
        assert gaz.label_for(lat, lon).city == "New York City"
    # The k nearest places alone would be New Jersey / Westchester suburbs; the exact radius
    # query is what keeps this right, so it must work even with a tiny k.
    monkeypatch.setattr(geo, "_NEAREST_K", 2)
    for lat, lon in points:
        assert gaz.label_for(lat, lon).city == "New York City"
    monkeypatch.setattr(geo, "_NEAREST_K", 64)
    assert gaz.label_for(*points[0]).city == "New York City"


def test_tiny_labelable_set_skips_the_radius_fallback():
    few = OfflineGazetteer(min_population=12_000_000)
    assert 0 < len(few._lab_rows) < geo._NEAREST_K
    assert few.label_for(31.2304, 121.4737).city == "Shanghai"


def test_place_containing_the_point_beats_a_bigger_one_that_also_does(gaz):
    # 2.0 km from Santa Clara's centre (inside its footprint) and 3.7 km from San Jose's
    # (also inside): the nearer place wins even though distance / footprint prefers San Jose.
    assert gaz.label_for(37.34875, -121.93365).city == "Santa Clara"
    assert gaz.label_for(37.3384, -121.8870).city == "San Jose"


def test_city_state_districts_are_absorbed(gaz):
    labelled = {lb.city for lb in gaz._lab_labels if lb.country_code == "HK"}
    assert "Hong Kong" in labelled
    assert not {"Jordan", "Wan Chai", "Kowloon", "Mong Kok"} & labelled


def test_label_many_rejects_out_of_range_coordinates(gaz):
    assert gaz.label_many([48.8584], [182.2945]) == [None]
    assert gaz.label_many([91.0], [2.2945]) == [None]
    assert gaz.label_many([-90.5], [2.2945]) == [None]
    assert gaz.label_many([48.8584], [-180.5]) == [None]


def test_max_city_km_caps_reach(gaz):
    # 8 km east of Key West, in the sea: a city by default, only the state when capped at 5 km
    assert gaz.label_for(24.5551, -81.70).city == "Key West"
    capped = OfflineGazetteer(max_city_km=5.0, max_country_km=250.0).label_for(24.5551, -81.70)
    assert (capped.city, capped.region, capped.country) == (None, "Florida", "United States")


def test_resolve_unaccented_primary_name(gaz):
    assert gaz.resolve("Modling").city == "Mödling"
    assert gaz.resolve("modling").region == "Lower Austria"


# --- alternate names, nicknames and exonyms


def test_alternate_name_keys_are_normalised_before_the_length_check(gaz):
    # Réo (Burkina Faso) has the alternate name "ri'o"; it must not become the key "rio".
    reo = [i for i in gaz._core.alt.get("rio", []) if gaz._core.names[i] == "Réo"]
    assert not reo
    assert all(len(key) >= 4 for key in gaz._core.alt)
    assert gaz.resolve("Rio").city == "Rio de Janeiro"
    assert gaz.resolve("Rio").country == "Brazil"


@pytest.mark.parametrize(
    ("query", "city", "country"),
    [
        ("Rio", "Rio de Janeiro", "Brazil"),
        ("Calcutta", "Kolkata", "India"),
        ("Bombay", "Mumbai", "India"),
        ("Madras", "Chennai", "India"),
        ("Bangalore", "Bengaluru", "India"),
        ("Peking", "Beijing", "China"),
        ("Saigon", "Ho Chi Minh City", "Vietnam"),
        ("Cracow", "Kraków", "Poland"),
        ("Munich", "Munich", "Germany"),
        ("Naples", "Naples", "Italy"),
        ("Florence", "Florence", "Italy"),
        ("Venice", "Venice", "Italy"),
        ("Rome", "Rome", "Italy"),
        ("Vienna", "Vienna", "Austria"),
        ("Constantinople", "Istanbul", "Turkey"),
        ("Cochin", "Kochi", "India"),
        ("Trivandrum", "Thiruvananthapuram", "India"),
        ("Benares", "Varanasi", "India"),
        ("Cologne", "Köln", "Germany"),
        ("Seville", "Sevilla", "Spain"),
    ],
)
def test_nicknames_and_exonyms_beat_tiny_namesakes(gaz, query, city, country):
    found = gaz.resolve(query)
    assert (found.city, found.country) == (city, country)


def test_nickname_rule_prefers_a_far_bigger_alternate_but_not_a_similar_one(gaz):
    # "Calcutta, South Africa" is a real hamlet, still reachable by qualifier.
    assert gaz.resolve("Calcutta, South Africa") is None or (
        gaz.resolve("Calcutta, South Africa").country == "South Africa"
    )
    # same-size alternates must not outrank a primary-name match (Paris stays Paris)
    assert gaz.resolve("Paris").city == "Paris" and gaz.resolve("Paris").country == "France"
    for alias in geo._CITY_ALIASES:
        found = gaz.resolve(alias)
        assert found is not None and (found.city or found.near), alias


def test_cote_divoire_spellings(gaz):
    for query in ("Côte d'Ivoire", "Cote d'Ivoire", "cote d ivoire", "Ivory Coast"):
        found = gaz.resolve(query)
        assert (found.country, found.city) == ("Ivory Coast", None), query


# --- abbreviations


@pytest.mark.parametrize(
    ("query", "city", "region", "country"),
    [
        ("Vancouver, BC", "Vancouver", "British Columbia", "Canada"),
        ("Sydney, NSW", "Sydney", "New South Wales", "Australia"),
        ("Toronto, ON", "Toronto", "Ontario", "Canada"),
        ("Calgary, AB", "Calgary", "Alberta", "Canada"),
        ("Perth, WA", "Perth", "Western Australia", "Australia"),
        ("Melbourne, VIC", "Melbourne", "Victoria", "Australia"),
        ("Los Angeles, CA, USA", "Los Angeles", "California", "United States"),
        ("Seattle, WA, USA", "Seattle", "Washington", "United States"),
        ("Portland, OR", "Portland", "Oregon", "United States"),
    ],
)
def test_region_abbreviations_as_qualifiers(gaz, query, city, region, country):
    found = gaz.resolve(query)
    assert (found.city, found.region, found.country) == (city, region, country)


def test_region_abbreviations_on_their_own(gaz):
    assert gaz.resolve("BC").region == "British Columbia"
    assert gaz.resolve("NSW").region == "New South Wales"
    assert gaz.resolve("TX").region == "Texas"
    assert gaz.resolve("CA").country == "Canada"  # a country code wins over California
    assert gaz.resolve("bc") is None  # lower case is not taken for an abbreviation


# --- regions beyond the US and curated destinations


@pytest.mark.parametrize(
    ("query", "region", "country"),
    [
        ("Rajasthan", "Rajasthan", "India"),
        ("Karnataka", "Karnataka", "India"),
        ("Tamil Nadu", "Tamil Nadu", "India"),
        ("Kerala", "Kerala", "India"),
        ("Goa", "Goa", "India"),
        ("Kashmir", "Jammu and Kashmir", "India"),
        ("Bavaria", "Bavaria", "Germany"),
        ("Bayern", "Bavaria", "Germany"),
        ("Tuscany", "Tuscany", "Italy"),
        ("Toscana", "Tuscany", "Italy"),
        ("Sicily", "Sicily", "Italy"),
        ("Andalusia", "Andalusia", "Spain"),
        ("Andalucía", "Andalusia", "Spain"),
        ("Catalonia", "Catalonia", "Spain"),
        ("Provence", "Provence-Alpes-Côte d'Azur", "France"),
        ("Brittany", "Brittany", "France"),
        ("Tyrol", "Tyrol", "Austria"),
        ("British Columbia", "British Columbia", "Canada"),
    ],
)
def test_resolve_regions_beyond_the_us(gaz, query, region, country):
    found = gaz.resolve(query)
    assert (found.region, found.country, found.city) == (region, country, None)


def test_goa_is_the_indian_state_not_a_philippine_town(gaz):
    assert gaz.resolve("Goa").country == "India"
    assert gaz.label_for(15.5524, 73.7517).region == "Goa"  # Baga Beach: the filter matches
    assert gaz.resolve("Goa, Philippines").city == "Goa"


@pytest.mark.parametrize(
    "query",
    [
        "Cornwall",
        "Yorkshire",
        "Bali",
        "Mount Rainier",
        "Interlaken",
        "Lake Tahoe",
        "Banff",
        "Alsace",
        "Cape Cod",
        "Maui",
        "Machu Picchu",
        "Angkor Wat",
    ],
)
def test_notable_destinations_resolve_to_near_filters(gaz, query):
    found = gaz.resolve(query)
    assert found.near is not None and found.city is None and found.country is None
    assert found.text == query


def test_notable_destinations_win_over_namesake_towns_but_qualifiers_still_work(gaz):
    cornwall = gaz.resolve("Cornwall").near
    assert abs(cornwall[0] - 50.35) < 0.5 and abs(cornwall[1] + 4.85) < 0.5
    assert gaz.resolve("Cornwall, England").near == cornwall
    assert gaz.resolve("Cornwall, Ontario").city == "Cornwall"  # the Canadian city
    california = gaz.resolve("Interlaken, California").near  # the small US namesake
    assert abs(california[0] - 36.95) < 0.2 and abs(california[1] + 121.73) < 0.2
    assert gaz.resolve("Interlaken, Switzerland").near is not None
    maryland = gaz.resolve("Mount Rainier, Maryland").near  # the small town in Maryland
    assert abs(maryland[0] - 38.94) < 0.2 and abs(maryland[1] + 76.96) < 0.2
    assert gaz.resolve("Mount Rainier, Washington").near[:2] == (46.8523, -121.7603)
    assert gaz.resolve("Lake Tahoe, California").near is not None
    virginia = gaz.resolve("Yorkshire, Virginia").near  # the small US namesake
    assert abs(virginia[0] - 38.79) < 0.2 and abs(virginia[1] + 77.45) < 0.2
    # unknown places stay unknown, even next to curated ones
    assert gaz.resolve("Cornwall, Mars") is None
    assert gaz.resolve("Narnia") is None


def test_notable_table_is_sane(gaz):
    remote = {"Uluru"}  # the nearest town of 5 000 people is hundreds of km away
    tables = (gaz._core, gaz._wide_table())
    names = set()
    for entry_names, ccs, lat, lon, radius in geo._NOTABLE_PLACES:
        assert -90 <= lat <= 90 and -180 <= lon <= 180
        assert 0 < radius <= 500
        assert ccs and all(cc in gaz._countries for cc in ccs)
        for name in entry_names:
            key = geo._norm(name)
            assert key not in names or key in {"jasper", "banff"}, name  # no duplicate keys
            names.add(key)
        if entry_names[0] in remote:
            continue
        nearest = min(
            float(geo._haversine_km(lat, lon, table.lat, table.lon).min()) for table in tables
        )
        assert nearest <= 120, (entry_names[0], nearest)  # centre is in inhabited country
        # every spelling resolves to this very entry
        for name in entry_names:
            assert gaz.resolve(name).near == (lat, lon, radius), name


def test_region_aliases_point_at_existing_regions(gaz):
    for alias, region in geo._REGION_ALIASES.items():
        assert region in gaz._regions, alias
        assert gaz.resolve(alias).region == gaz._regions[region], alias
