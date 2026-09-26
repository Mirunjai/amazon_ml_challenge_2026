from __future__ import annotations

import re
import unicodedata

CORP_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "plc", "pvt", "private", "privatelimited",
    "llp", "gmbh", "sa", "sas", "spa",
}


def unicode_fold(text: object) -> str:
    if text is None:
        return ""
    value = str(text)
    value = unicodedata.normalize("NFKC", value)
    return value.casefold()


def normalize_text(text: object) -> str:
    """Conservative normalization; does not remove potentially useful tokens."""
    value = unicode_fold(text)
    value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
    value = re.sub(r"[^\w\s.-]", " ", value, flags=re.UNICODE)
    value = re.sub(r"[_]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_name(text: object, strip_legal_suffixes: bool = False) -> str:
    value = normalize_text(text)
    if not strip_legal_suffixes:
        return value

    tokens = value.split()
    while tokens and tokens[-1].replace(".", "") in CORP_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_address(text: object) -> str:
    value = normalize_text(text)
    # Keep numbers: they are often highly discriminative in addresses.
    value = re.sub(r"\broad\b", "rd", value)
    value = re.sub(r"\bstreet\b", "st", value)
    value = re.sub(r"\bavenue\b", "ave", value)
    value = re.sub(r"\baven\b", "ave", value)
    value = re.sub(r"\broadway\b", "roadway", value)
    return re.sub(r"\s+", " ", value).strip()


def token_set(text: str) -> set[str]:
    return {tok for tok in normalize_text(text).split() if tok}


def character_ngrams(text: str, n: int = 3) -> set[str]:
    value = normalize_text(text).replace(" ", "")
    if len(value) < n:
        return {value} if value else set()
    return {value[i:i+n] for i in range(len(value) - n + 1)}
