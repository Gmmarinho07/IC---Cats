CATALYST_ALIASES = {

    # Hydroxyapatite

    "hap": "hydroxyapatite",
    "hydroxyapatite": "hydroxyapatite",

    # Magnesium oxide

    "mgo": "magnesium oxide",
    "magnesia": "magnesium oxide",
    "magnesium oxide": "magnesium oxide",

    # Mg-Al systems

    "mgal-ldo": "mg-al mixed oxide",
    "mg-al mixed oxide": "mg-al mixed oxide",

    # Titanium oxide

    "tio2": "titanium dioxide",
    "titanium dioxide": "titanium dioxide",

    # Ruthenium

    "atomic ru": "ru",
    "ru": "ru"
}


def normalize(text):

    if text is None:
        return None

    text = str(text).lower().strip()

    return CATALYST_ALIASES.get(
        text,
        text
    )


def normalize_list(values):

    if not values:
        return []

    return [
        normalize(v)
        for v in values
    ]