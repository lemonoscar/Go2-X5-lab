#!/usr/bin/env python3
"""Download and register local Layer 1 PBR floor materials and HDRI assets."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "source" / "robot_lab" / "data"
MANIFEST_PATH = DATA_ROOT / "Materials" / "manifests" / "vla_render_asset_manifest.json"
REGISTRY_PATH = DATA_ROOT / "Materials" / "manifests" / "vla_render_asset_registry.json"
FLOOR_ROOT = DATA_ROOT / "Materials" / "Floors"
HDRI_ROOT = DATA_ROOT / "HDRI"
API_URL = "https://ambientcg.com/api/v2/full_json"
USER_AGENT = "Go2-X5-lab/1.0"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download_file(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def fetch_asset_payload(asset_id: str) -> dict:
    url = f"{API_URL}?id={asset_id}&include=downloadData,imageData"
    payload = fetch_json(url)
    found_assets = payload.get("foundAssets", [])
    if not found_assets:
        raise RuntimeError(f"ambientCG did not return data for asset '{asset_id}'.")
    return found_assets[0]


def choose_download_entry(asset_payload: dict, download_attribute: str) -> dict:
    folders = asset_payload.get("downloadFolders", {})
    for folder in folders.values():
        categories = folder.get("downloadFiletypeCategories", {})
        zip_downloads = categories.get("zip", {}).get("downloads", [])
        for entry in zip_downloads:
            if str(entry.get("attribute", "")).lower() == download_attribute.lower():
                return entry
    raise RuntimeError(
        f"Could not find download attribute '{download_attribute}' for asset '{asset_payload.get('assetId')}'."
    )


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.is_dir() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_downloaded_asset(asset_id: str, download_attribute: str, output_dir: Path, force: bool) -> None:
    ensure_clean_dir(output_dir, force=force)
    if not force and any(output_dir.iterdir()):
        return

    asset_payload = fetch_asset_payload(asset_id)
    download_entry = choose_download_entry(asset_payload, download_attribute)
    archive_path = output_dir / download_entry["fileName"]

    print(f"[DOWNLOAD] {asset_id} -> {download_entry['fileName']}")
    download_file(download_entry["downloadLink"], archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    archive_path.unlink()


def _candidate_rank(path: Path, preferred_tokens: tuple[str, ...]) -> tuple[int, str]:
    name = path.name.lower()
    for index, token in enumerate(preferred_tokens):
        if token in name:
            return index, name
    return len(preferred_tokens), name


def pick_first_match(asset_dir: Path, include_tokens: tuple[str, ...], suffixes: tuple[str, ...]) -> str | None:
    candidates = []
    for path in asset_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not name.endswith(suffixes):
            continue
        if not all(token in name for token in include_tokens):
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: _candidate_rank(path, include_tokens))
    return str(candidates[0].relative_to(DATA_ROOT))


def scan_floor_maps(asset_dir: Path) -> dict:
    return {
        "color_texture": pick_first_match(asset_dir, ("basecolor",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("color",), (".jpg", ".jpeg", ".png")),
        "roughness_texture": pick_first_match(asset_dir, ("roughness",), (".jpg", ".jpeg", ".png")),
        "normal_texture": pick_first_match(asset_dir, ("normaldx",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("normalgl",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("normal",), (".jpg", ".jpeg", ".png")),
        "metallic_texture": pick_first_match(asset_dir, ("metalness",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("metallic",), (".jpg", ".jpeg", ".png")),
        "ao_texture": pick_first_match(asset_dir, ("ambientocclusion",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("_ao",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("-ao",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("occlusion",), (".jpg", ".jpeg", ".png")),
        "orm_texture": pick_first_match(asset_dir, ("_orm",), (".jpg", ".jpeg", ".png"))
        or pick_first_match(asset_dir, ("-orm",), (".jpg", ".jpeg", ".png")),
    }


def scan_hdri_map(asset_dir: Path) -> str | None:
    for suffix in (".hdr", ".exr"):
        candidates = sorted(path for path in asset_dir.rglob("*") if path.is_file() and path.suffix.lower() == suffix)
        if candidates:
            return str(candidates[0].relative_to(DATA_ROOT))
    return None


def build_registry(manifest: dict) -> dict:
    registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(MANIFEST_PATH.relative_to(DATA_ROOT)),
        "floor_materials": {},
        "hdri_profiles": {},
    }

    for category, assets in manifest.get("floor_materials", {}).items():
        registry["floor_materials"][category] = []
        for asset_cfg in assets:
            asset_id = asset_cfg["asset_id"]
            asset_dir = FLOOR_ROOT / category / asset_id
            floor_maps = scan_floor_maps(asset_dir)
            if floor_maps["color_texture"] is None:
                print(f"[WARN] Missing color texture for {asset_id}; skipping registry entry.")
                continue

            registry["floor_materials"][category].append(
                {
                    "asset_id": asset_id,
                    "asset_dir": str(asset_dir.relative_to(DATA_ROOT)),
                    **floor_maps,
                    "texture_scale": asset_cfg.get("texture_scale", [0.25, 0.25]),
                    "bump_factor": asset_cfg.get("bump_factor", 1.0),
                    "ao_to_diffuse": asset_cfg.get("ao_to_diffuse", 0.25),
                }
            )

    for profile_name, profile_cfg in manifest.get("hdri_profiles", {}).items():
        asset_dir = HDRI_ROOT / profile_name
        texture_file = scan_hdri_map(asset_dir)
        if texture_file is None:
            print(f"[WARN] Missing HDRI texture for {profile_name}; skipping registry entry.")
            continue

        registry["hdri_profiles"][profile_name] = {
            "asset_id": profile_cfg["asset_id"],
            "asset_dir": str(asset_dir.relative_to(DATA_ROOT)),
            "texture_file": texture_file,
            "intensity": profile_cfg.get("intensity", 1200.0),
            "fallback_color": profile_cfg.get("fallback_color", [0.75, 0.75, 0.75]),
            "visible_in_primary_ray": profile_cfg.get("visible_in_primary_ray", True),
            "texture_format": "automatic",
        }

    return registry


def download_requested_assets(manifest: dict, only: str, force: bool, skip_download: bool) -> None:
    if skip_download:
        return

    if only in {"all", "floors"}:
        for category, assets in manifest.get("floor_materials", {}).items():
            for asset_cfg in assets:
                ensure_downloaded_asset(
                    asset_id=asset_cfg["asset_id"],
                    download_attribute=asset_cfg["download_attribute"],
                    output_dir=FLOOR_ROOT / category / asset_cfg["asset_id"],
                    force=force,
                )

    if only in {"all", "hdri"}:
        for profile_name, profile_cfg in manifest.get("hdri_profiles", {}).items():
            ensure_downloaded_asset(
                asset_id=profile_cfg["asset_id"],
                download_attribute=profile_cfg["download_attribute"],
                output_dir=HDRI_ROOT / profile_name,
                force=force,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local Layer 1 PBR floor and HDRI assets.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--only", choices=["all", "floors", "hdri"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract assets.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip network download and only rebuild the local registry from existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    FLOOR_ROOT.mkdir(parents=True, exist_ok=True)
    HDRI_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_requested_assets(manifest, only=args.only, force=args.force, skip_download=args.skip_download)
    registry = build_registry(manifest)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

    floor_counts = {key: len(value) for key, value in registry["floor_materials"].items()}
    print(f"[DONE] Wrote registry to {REGISTRY_PATH}")
    print(f"[DONE] Floor assets: {floor_counts}")
    print(f"[DONE] HDRI profiles: {sorted(registry['hdri_profiles'].keys())}")


if __name__ == "__main__":
    main()
