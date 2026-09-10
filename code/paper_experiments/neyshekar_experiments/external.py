"""Held-out external corpora with pinned sources, audio hashes and explicit segmentation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .manifests import freeze
from .protocol import DATA, ROOT, SAMPLE_RATE, digest, file_digest

PSRB_REVISION = "626745e790667d6cbf70bdb10a260d3af90ed2d2"
PSRB_SOURCE = "https://huggingface.co/datasets/PartAI/PSRB"
YOUTUBE_SOURCE = "https://github.com/ReihanehIranManesh/persian-youtube-whisper-benchmark"


def psrb_rows(labels):
    """Correct the documented header/row permutation at the pinned PSRB revision.

    Header: path,duration,speaker-count,text; actual rows: path,text,duration,count.
    Reject unexpected layouts instead of evaluating numeric strings as references.
    """
    with Path(labels).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header[:4] != ["audio_path", "audio_duration", "number_of_speakers", "text"]:
            raise ValueError("Unexpected PSRB header; review the pinned source adapter")
        corrected = ["audio_path", "text", "audio_duration", "number_of_speakers", *header[4:]]
        out = []
        for values in reader:
            if len(values) != len(header):
                raise ValueError("Malformed PSRB CSV row")
            row = dict(zip(corrected, values))
            float(row["audio_duration"])
            int(row["number_of_speakers"])
            if not re.search(r"[\u0600-\u06ff]", row["text"]):
                raise ValueError("PSRB transcript is not in the expected column")
            row["id"] = row["audio_path"]
            # Original recording/speaker grouping is unavailable in the public labels.
            out.append(row)
    return out


def freeze_external(name, rows, audio_root, provenance):
    import soundfile as sf

    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Evaluation name must contain lowercase letters/digits/underscores")
    audio_root = Path(audio_root).resolve()
    if not audio_root.is_relative_to(ROOT.resolve()):
        raise ValueError("Keep external audio inside the repository so manifests remain portable")
    records, hashes = [], {}
    for row in rows:
        row = dict(row)
        path = (audio_root / row["audio_path"]).resolve()
        if not path.is_relative_to(audio_root):
            raise ValueError("Audio path escapes the dataset directory")
        info = sf.info(path)
        start = float(row.get("start_seconds", 0))
        end = float(row.get("end_seconds", info.duration))
        if (
            not np.isfinite([start, end]).all()
            or not 0 <= start < end <= info.duration + 1 / info.samplerate
        ):
            raise ValueError(f"Invalid segment boundaries: {row['id']}")
        if not str(row["text"]).strip():
            raise ValueError("Empty external reference")
        hashes.setdefault(str(path), None)
        if hashes[str(path)] is None:
            hashes[str(path)] = file_digest(path)
        row.update(
            id=str(row["id"]),
            corpus=name,
            split="test",
            duration=end - start,
            start_seconds=start,
            end_seconds=end,
            audio_path=str(path.relative_to(audio_root)),
            audio_sha256=hashes[str(path)],
        )
        records.append(row)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty or duplicate external evaluation IDs")
    payload = {
        "name": name,
        "audio_root": audio_root.relative_to(ROOT.resolve()).as_posix(),
        "source": provenance,
        "segmentation": provenance["segmentation"],
        "records": records,
        "clips": len(records),
        "hours": sum(r["duration"] for r in records) / 3600,
    }
    payload["sha256"] = digest(payload)
    freeze(DATA / "external/manifests" / f"{name}.json", payload)
    return payload


def prepare_psrb(download=False):
    root = DATA / "external/psrb"
    if download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "PartAI/PSRB",
            repo_type="dataset",
            revision=PSRB_REVISION,
            local_dir=root,
            allow_patterns=["Labels.csv", "Files/*", "README.md"],
        )
    return freeze_external(
        "psrb_sample",
        psrb_rows(root / "Labels.csv"),
        root,
        {
            "url": PSRB_SOURCE,
            "revision": PSRB_REVISION,
            "labels_sha256": file_digest(root / "Labels.csv"),
            "segmentation": "released clips; no further segmentation",
        },
    )


def timestamp_rows(transcript, audio_path, recording_id, end_seconds):
    """Read the benchmark's alternating timestamp/text CSV; retain the final segment."""
    lines = []
    with Path(transcript).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if row and any(part.strip() for part in row):
                lines.append(",".join(row).strip().strip("`"))
    if len(lines) % 2:
        raise ValueError("Expected alternating timestamp/text lines")
    times, texts = [], []
    for timestamp, text in zip(lines[::2], lines[1::2]):
        if not re.fullmatch(r"\d{1,3}:\d{2}(?::\d{2})?", timestamp):
            raise ValueError(f"Invalid timestamp: {timestamp}")
        pieces = [int(p) for p in timestamp.split(":")]
        if any(p >= 60 for p in pieces[1:]):
            raise ValueError("Invalid timestamp component")
        seconds = 0
        for p in pieces:
            seconds = seconds * 60 + p
        times.append(seconds)
        texts.append(text)
    ends = [*times[1:], float(end_seconds)]
    if not times or any(a >= b for a, b in zip(times, ends)):
        raise ValueError("Timestamps must increase and precede recording end")
    return [
        {
            "id": f"{recording_id}:{i:05d}",
            "recording_id": recording_id,
            "audio_path": audio_path,
            "text": text,
            "start_seconds": start,
            "end_seconds": end,
        }
        for i, (text, start, end) in enumerate(zip(texts, times, ends))
    ]


def prepare_youtube(episodes, audio_root, revision):
    """episodes CSV: recording_id,audio_path,transcript_path; paths relative to audio_root.

    Supply the checked-out upstream commit and the authors' original audio. This
    oracle-boundary condition must not be pooled with silence-segmented results.
    """
    import soundfile as sf

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Provide the full upstream Git commit, not a moving branch")
    root = Path(audio_root).resolve()
    rows, transcripts = [], {}
    for episode in pd.read_csv(episodes, dtype=str, keep_default_na=False).to_dict("records"):
        audio = (root / episode["audio_path"]).resolve()
        transcript = (root / episode["transcript_path"]).resolve()
        if not audio.is_relative_to(root) or not transcript.is_relative_to(root):
            raise ValueError("Episode paths must be relative to audio_root")
        transcripts[episode["transcript_path"]] = file_digest(transcript)
        rows.extend(
            timestamp_rows(
                transcript, episode["audio_path"], episode["recording_id"], sf.info(audio).duration
            )
        )
    return freeze_external(
        "youtube_timestamps",
        rows,
        root,
        {
            "url": YOUTUBE_SOURCE,
            "revision": revision,
            "transcript_sha256": transcripts,
            "episodes_sha256": file_digest(Path(episodes)),
            "segmentation": "reference timestamps; final interval ends at audio EOF",
        },
    )


def resolve_audio_root(manifest):
    relative = Path(manifest["audio_root"])
    if relative.is_absolute():
        raise ValueError("Legacy absolute audio_root; regenerate the manifest from source data")
    resolved = (ROOT / relative).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("External audio_root escapes the repository")
    return resolved


class ExternalSpeech:
    def __init__(self, manifest):
        self.rows = pd.DataFrame(manifest["records"])
        self.provenance = {k: v for k, v in manifest.items() if k not in ("records", "audio_root")}
        self.audio_root = resolve_audio_root(manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import soundfile as sf

        row = self.rows.iloc[index]
        with sf.SoundFile(self.audio_root / row.audio_path) as stream:
            sr = stream.samplerate
            start, end = round(row.start_seconds * sr), round(row.end_seconds * sr)
            stream.seek(start)
            audio = stream.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return {"audio": audio, "text": row.text}


def external_sets(names):
    sets = {}
    for name in names:
        if name not in ("psrb_sample", "youtube_timestamps"):
            raise ValueError("Unknown external evaluation")
        manifest = json.loads((DATA / "external/manifests" / f"{name}.json").read_text())
        if digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
            raise ValueError("External manifest changed")
        audio_root = resolve_audio_root(manifest)
        checked = set()
        for row in manifest["records"]:
            path = (audio_root / row["audio_path"]).resolve()
            if not path.is_relative_to(audio_root):
                raise ValueError("Audio path escapes the dataset directory")
            if path not in checked:
                if file_digest(path) != row["audio_sha256"]:
                    raise ValueError(f"External audio changed: {path}")
                checked.add(path)
        sets[name] = ExternalSpeech(manifest)
    return sets
