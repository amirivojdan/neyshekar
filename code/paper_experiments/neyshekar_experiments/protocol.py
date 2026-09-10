"""Versioned constants, scoring, and artifact fingerprints (no model loading)."""

from __future__ import annotations

import hashlib
import json
import string
import unicodedata
from functools import lru_cache
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent  # Repository root, independent of the process working directory.
DATA = ROOT / "data"
PROTOCOL = "v2"
DATA_SEED = 42  # Never change this to repeat an optimization seed.
HF_REPO = "shekar-ai/neyshekar-v6-persian-asr-fa"
HF_REVISION = "7613a5adebabb5f8f1255a47ef42ca7d7b44046c"
CV_DIR = DATA / "cv26/cv-corpus-26.0-2026-06-12/fa"
MODEL_REVISIONS = {
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "facebook/wav2vec2-xls-r-300m": "1a640f32ac3e39899438a2931f9924c02f080a54",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "facebook/mms-1b-all": "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
}
SCORING = "shekar-1.6.3; digit scripts unified; bidi controls removed; ZWNJ/punctuation to spaces; numbers not verbalized"
SAMPLE_RATE = 16_000
MAX_TRAIN_SECONDS = 20.0  # Common eligibility for BOTH architectures, before matching.
EFFECTIVE_BATCH = 64
EPOCHS = 3


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment_spec_hashes():
    """Include exact dependency inputs and available platform locks in run identity."""
    return {
        path.name: file_digest(path)
        for path in sorted(CODE.glob("requirements*"))
        if path.is_file() and path.suffix in (".txt", ".lock")
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@lru_cache(maxsize=1)
def text_tools():
    from shekar import Normalizer, WordTokenizer

    return Normalizer(), WordTokenizer()


def normalise_for_scoring(text: str) -> str:
    normalizer, tokenizer = text_tools()
    digits = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    # Directional formatting controls are not spoken characters. Apply the same
    # removal to every corpus, CTC target, reference, and hypothesis.
    bidi_controls = "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
    text = str(text).translate({ord(char): None for char in bidi_controls})
    text = normalizer(text).translate(digits).replace("\u200c", " ")
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(tokenizer.tokenize(text))


def fixed_ctc_vocabulary() -> dict[str, int]:
    """A priori alphabet, independent of corpora, manifests, and evaluation text.

    Arabic-block letters/marks cover Persian and orthographic variants. ASCII
    letters cover code-switching; digits have the same support in every arm.
    A soft hyphen and common non-punctuation symbols are retained by the legacy
    scoring normalizer, so they have explicit support too. Unknowns remain UNK.
    """
    chars = {chr(i) for i in range(0x0600, 0x0700) if unicodedata.category(chr(i))[0] in {"L", "M"}}
    chars.update(string.ascii_letters + string.digits + "=+×÷°\u00ad")
    vocabulary = {c: i for i, c in enumerate(sorted(chars))}
    for special in ("|", "[UNK]", "[PAD]"):
        vocabulary[special] = len(vocabulary)
    return vocabulary


def score(references, hypotheses) -> dict:
    from jiwer import process_characters, process_words

    if len(references) != len(hypotheses):
        raise ValueError("Reference/hypothesis length mismatch")
    pairs = [
        (normalise_for_scoring(r), normalise_for_scoring(h)) for r, h in zip(references, hypotheses)
    ]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        raise ValueError("No nonempty scoring references")
    refs, hyps = map(list, zip(*pairs))
    return {
        "n": len(refs),
        "n_total": len(references),
        "n_scored": len(refs),
        "n_empty_reference": len(references) - len(refs),
        "wer": 100 * process_words(refs, hyps).wer,
        "cer": 100 * process_characters(refs, hyps).cer,
    }
