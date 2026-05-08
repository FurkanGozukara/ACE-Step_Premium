"""Display labels for language-code dropdowns."""

from __future__ import annotations

from acestep.constants import VALID_LANGUAGES


LANGUAGE_NAMES = {
    "ar": "Arabic",
    "az": "Azerbaijani",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "la": "Latin",
    "lt": "Lithuanian",
    "ms": "Malay",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sk": "Slovak",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "unknown": "Instrumental / auto",
}


def language_dropdown_choices() -> list[tuple[str, str]]:
    """Return dropdown choices as ``(display_name, language_code)`` pairs."""

    return [(LANGUAGE_NAMES.get(code, code), code) for code in VALID_LANGUAGES]
