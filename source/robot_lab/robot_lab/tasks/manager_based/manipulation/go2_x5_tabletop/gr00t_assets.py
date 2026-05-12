# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Small resolver for optional GR00T tabletop USD assets.

The tabletop task must remain self-contained when the sibling GR00T checkout is
absent or when its Git LFS assets have not been fetched. This module only
returns paths that are present and look like real USD/USDA payloads.
"""

from __future__ import annotations

from pathlib import Path


GR00T_REPO_NAME = "GR00T-VisualSim2Real"
GO2_X5_REPO_NAME = "Go2-X5-lab"

GR00T_ASSETS: dict[str, str] = {
    "simple_table_new": "gr00t/rl/data/objects/simple/table_new.usd",
    "simple_table": "gr00t/rl/data/objects/simple/table.usd",
    "simple_tray": "gr00t/rl/data/objects/simple/tray-28-28-10-thick-15mm.usd",
    "simple_bottle": "gr00t/rl/data/objects/simple/bottle.usd",
    "simple_cube": "gr00t/rl/data/objects/simple/cube.usd",
    "grab_table": "gr00t/rl/data/objects/grab/grab_table.usda",
    "grab_apple": "gr00t/rl/data/objects/grab/apple.usda",
    "grab_bottle": "gr00t/rl/data/objects/grab/bottle.usd",
    "grab_bowl": "gr00t/rl/data/objects/grab/bowl.usda",
    "grab_coffeemug": "gr00t/rl/data/objects/grab/coffeemug.usda",
    "grab_mug": "gr00t/rl/data/objects/grab/mug.usda",
    "grab_rubberduck": "gr00t/rl/data/objects/grab/rubberduck.usda",
    "grab_waterbottle": "gr00t/rl/data/objects/grab/waterbottle.usda",
}


def find_workspace_root() -> Path | None:
    """Return the parent directory containing both sibling repositories."""

    for parent in Path(__file__).resolve().parents:
        if (parent / GO2_X5_REPO_NAME).is_dir() and (parent / GR00T_REPO_NAME).is_dir():
            return parent
    return None


def get_gr00t_repo_root() -> Path | None:
    workspace_root = find_workspace_root()
    if workspace_root is None:
        return None
    repo_root = workspace_root / GR00T_REPO_NAME
    return repo_root if repo_root.is_dir() else None


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            header = file.read(128)
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def get_gr00t_asset_path(name: str) -> str | None:
    """Return an absolute USD/USDA asset path, or ``None`` if unavailable."""

    relative_path = GR00T_ASSETS.get(name)
    repo_root = get_gr00t_repo_root()
    if relative_path is None or repo_root is None:
        return None

    asset_path = repo_root / relative_path
    if not asset_path.is_file() or _is_git_lfs_pointer(asset_path):
        return None
    return str(asset_path)


def get_first_gr00t_asset_path(names: list[str] | tuple[str, ...]) -> str | None:
    """Return the first available asset from an ordered candidate list."""

    for name in names:
        asset_path = get_gr00t_asset_path(name)
        if asset_path is not None:
            return asset_path
    return None

