"""Download LFW (free public research dataset) and build reproducible open-set evaluation splits.

    python scripts/prepare_lfw.py

Creates (images are copied; data/ is git-ignored, nothing here is committed):

    data/lfw_eval/val/   enrolled/<id>/*.jpg        gallery images (3 per known identity)
                         test/known/<id>/*.jpg      other photos of enrolled identities (<= 5)
                         test/unknown/<id>/*.jpg    identities that are NEVER enrolled
    data/lfw_eval/test/  same structure, completely different people

The validation split is used to choose thresholds; the test split is used only to report
results with those thresholds (no person appears in both splits).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Same mirror + checksum that scikit-learn's fetch_lfw_people uses.
LFW_URL = "https://ndownloader.figshare.com/files/5976018"
LFW_SHA256 = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
DEFAULT_LFW_HOME = Path.home() / "scikit_learn_data" / "lfw_home"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_lfw(lfw_home: Path) -> Path:
    images_dir = lfw_home / "lfw"
    if images_dir.is_dir() and any(images_dir.iterdir()):
        return images_dir
    lfw_home.mkdir(parents=True, exist_ok=True)
    archive = lfw_home / "lfw.tgz"
    if not archive.exists():
        print(f"Downloading LFW (~173 MB) to {archive} ...")
        urllib.request.urlretrieve(LFW_URL, archive)
    digest = sha256(archive)
    if digest != LFW_SHA256:
        raise SystemExit(f"Checksum mismatch for {archive} ({digest}); delete it and retry.")
    print("Checksum OK, extracting ...")
    with tarfile.open(archive) as tar:
        tar.extractall(lfw_home)
    return images_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lfw-home", type=Path, default=DEFAULT_LFW_HOME)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "lfw_eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gallery-per-person", type=int, default=3)
    ap.add_argument("--max-known-probes", type=int, default=5)
    ap.add_argument("--max-unknown-probes", type=int, default=4)
    ap.add_argument("--singles-per-split", type=int, default=1000,
                    help="extra one-photo people added as unknowns to each split")
    args = ap.parse_args()

    images_dir = ensure_lfw(args.lfw_home)
    rng = random.Random(args.seed)
    people = {d.name: sorted(p.name for p in d.glob("*.jpg")) for d in sorted(images_dir.iterdir()) if d.is_dir()}

    multi = sorted(n for n, imgs in people.items() if len(imgs) >= args.gallery_per_person + 1)
    singles = sorted(n for n, imgs in people.items() if len(imgs) == 1)
    rng.shuffle(multi)
    rng.shuffle(singles)

    half = len(multi) // 2
    splits = {"val": multi[:half], "test": multi[half:]}
    single_splits = {"val": singles[:args.singles_per_split],
                     "test": singles[args.singles_per_split:2 * args.singles_per_split]}

    if args.out.exists():
        shutil.rmtree(args.out)
    manifest = {"seed": args.seed, "source": "LFW (lfw.tgz, sha256 verified)", "splits": {}}
    for split, names in splits.items():
        root = args.out / split
        known, unknown = names[: len(names) // 2], names[len(names) // 2:]
        stats = {"known_identities": len(known), "gallery_images": 0, "known_probes": 0,
                 "unknown_identities": 0, "unknown_probes": 0}
        for name in known:
            imgs = people[name][:]
            rng.shuffle(imgs)
            gallery, probes = imgs[:args.gallery_per_person], imgs[args.gallery_per_person:][:args.max_known_probes]
            for sub, files in (("enrolled", gallery), ("test/known", probes)):
                dst = root / sub / name
                dst.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(images_dir / name / f, dst / f)
            stats["gallery_images"] += len(gallery)
            stats["known_probes"] += len(probes)
        for name in unknown + single_splits[split]:
            imgs = people[name][:]
            rng.shuffle(imgs)
            dst = root / "test/unknown" / name
            dst.mkdir(parents=True, exist_ok=True)
            for f in imgs[:args.max_unknown_probes]:
                shutil.copy2(images_dir / name / f, dst / f)
            stats["unknown_identities"] += 1
            stats["unknown_probes"] += min(len(imgs), args.max_unknown_probes)
        manifest["splits"][split] = stats
        print(split, stats)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
