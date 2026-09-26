from __future__ import annotations

import re
import unicodedata
import math
import numbers
import pandas as pd


# Legal/business suffixes that can be removed when needed
CORP_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "plc",
    "pvt",
    "private",
    "privatelimited",
    "llp",
    "gmbh",
    "sa",
    "sas",
    "spa",
}
COUNTRY_ALIASES = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",

    "india": "India",
    "in": "India",
}



def unicode_fold(text: object) -> str:
    """
    Normalize Unicode characters and lowercase text.

    Missing values are returned as an empty string.
    """
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    value = str(text)
    value = unicodedata.normalize("NFKC", value).casefold()

    return value

def normalize_country(text: object) -> str:
    """
    Normalize country values conservatively.

    Missing country values remain empty.
    """

    value = unicode_fold(text)

    if not value:
        return ""

    return COUNTRY_ALIASES.get(value, value)


def normalize_text(text: object) -> str:
    """
    Conservative Unicode-aware text normalization.

    Preserves letters, numbers, and Unicode combining marks
    so non-English names are not damaged.
    """

    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    value = str(text)

    # Unicode normalization + lowercase
    value = unicodedata.normalize("NFKC", value).casefold()

    result = []

    for char in value:
        category = unicodedata.category(char)

        # Keep letters, numbers, and combining marks
        if (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("M")
        ):
            result.append(char)

        # Keep spaces
        elif char.isspace():
            result.append(" ")

        # Keep period and hyphen
        elif char in ".-":
            result.append(char)

        # Convert other punctuation/symbols to spaces
        else:
            result.append(" ")

    value = "".join(result)

    # Collapse repeated whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value

def normalize_name(
    text: object,
    strip_legal_suffixes: bool = False,
) -> str:
    """
    Normalize a business name.

    Optionally removes legal/company suffixes from the end.
    """

    value = normalize_text(text)

    if not value:
        return ""

    if strip_legal_suffixes:
        words = value.split()

        # Remove legal suffixes from the end
        while words and words[-1].replace(".", "") in CORP_SUFFIXES:
            words.pop()

        value = " ".join(words)

    return value


def normalize_address(text: object) -> str:
    """
    Conservative business-address normalization.

    Keeps numbers because they are highly useful for distinguishing
    different addresses.
    """

    value = normalize_text(text)

    # Common street-name abbreviations
    value = re.sub(r"\broad\b", "rd", value)
    value = re.sub(r"\bstreet\b", "st", value)
    value = re.sub(r"\bavenue\b", "ave", value)
    value = re.sub(r"\baven\b", "ave", value)

    # Do not accidentally change "Broadway" into "brdwy" etc.
    value = re.sub(r"\broadway\b", "roadway", value)

    # Final whitespace cleanup
    value = re.sub(r"\s+", " ", value).strip()

    return value


def token_set(text: str) -> set[str]:
    """
    Convert normalized text into a set of tokens.
    """

    return {
        token
        for token in normalize_text(text).split()
        if token
    }


def character_ngrams(
    text: str,
    n: int = 3
) -> set[str]:
    """
    Generate character n-grams.

    Example:
        "apple", n=3
        -> {"app", "ppl", "ple"}
    """

    value = normalize_text(text).replace(" ", "")

    if not value:
        return set()

    if len(value) < n:
        return {value}

    return {
        value[i:i+n]
        for i in range(len(value) - n + 1)
    }

def add_normalized_columns(df):
    """
    Add normalized fields required by the matching pipeline.

    Keeps the original raw columns unchanged.
    """
    out = df.copy()

    out["name_norm"] = out["business_name"].map(normalize_name)

    out["name_nosuffix"] = out["business_name"].map(
        lambda x: normalize_name(
            x,
            strip_legal_suffixes=True
        )
    )

    out["address_norm"] = out["business_address"].map(
        normalize_address
    )

    out["country_norm"] = out["country"].map(
        normalize_country
    )

    return out