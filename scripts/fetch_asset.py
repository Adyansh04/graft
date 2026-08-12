#!/usr/bin/env python3
"""Fetch a USD asset and everything it composes, mirroring the remote layout.

USD layers reference sublayers, payloads and textures by relative path, so a
flat download produces a stage that opens but contains nothing. This walks
the composition graph, fetching until the set closes.

Assets are gitignored — they are large binaries reproducible from this
script.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ISAAC_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.0/Isaac"
)

ASSETS = {
    # NVIDIA SimReady, CC BY 4.0. Native USD, split PBR maps, physics
    # preauthored. The development object.
    "mug": (
        f"{ISAAC_ROOT}/SimReady/Residential/Kitchen/Dishware/Coffee_Mug_A01",
        "sm_rc_dishware_mug_coffee_a01.usd",
    ),
    # YCB scan, CC BY 4.0. Ships one colour texture and no shading maps, so
    # its lighting is painted into the albedo. Kept as the fixture that
    # proves the baked-lighting warning fires — library assets never will.
    "ycb-mug": (f"{ISAAC_ROOT}/Props/YCB/Axis_Aligned", "025_mug.usd"),
}

TEXTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".exr", ".tga", ".hdr"}


def fetch(url: str, dest: Path) -> bool:
    if dest.is_file():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            dest.write_bytes(response.read())
        print(f"  fetched {dest.name}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"  MISSING {url} ({exc})", file=sys.stderr)
        return False


def layer_dependencies(path: Path) -> list[str]:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        return []
    deps = list(layer.GetCompositionAssetDependencies())
    # Texture paths live in attribute values rather than the composition
    # graph, so they need a separate sweep.
    for prim_spec_path in _walk_specs(layer):
        spec = layer.GetAttributeAtPath(prim_spec_path)
        if spec is None:
            continue
        value = spec.default
        for candidate in _asset_paths(value):
            if Path(candidate).suffix.lower() in TEXTURE_SUFFIXES:
                deps.append(candidate)
    return deps


def _walk_specs(layer):
    found = []
    layer.Traverse(layer.pseudoRoot.path, lambda p: found.append(p))
    return found


def _asset_paths(value):
    from pxr import Sdf

    if isinstance(value, Sdf.AssetPath):
        return [value.path] if value.path else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_asset_paths(item))
        return out
    return []


def fetch_closure(base_url: str, entry: str, dest_dir: Path) -> tuple[int, list[str]]:
    """Fetch the entry layer and everything reachable from it."""
    pending = [entry]
    seen: set[str] = set()
    missing: list[str] = []
    count = 0

    while pending:
        rel = _normalise(pending.pop())
        if rel is None or rel in seen:
            continue
        seen.add(rel)

        dest = dest_dir / rel
        existed = dest.is_file()
        if not fetch(f"{base_url}/{rel}", dest):
            missing.append(rel)
            continue
        if not existed:
            count += 1

        if dest.suffix.lower() in {".usd", ".usda", ".usdc"}:
            parent = Path(rel).parent
            for dep in layer_dependencies(dest):
                if dep.startswith(("http://", "https://", "omniverse://")):
                    continue
                pending.append((parent / dep).as_posix())

    return count, missing


def _normalise(rel: str) -> str | None:
    """Collapse ./ and ../ segments. None if the path escapes the asset root."""
    normalised = os.path.normpath(rel)
    if normalised.startswith("..") or os.path.isabs(normalised):
        print(f"  SKIP {rel} (outside the asset directory)", file=sys.stderr)
        return None
    return Path(normalised).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", nargs="?", default="mug", choices=sorted(ASSETS))
    parser.add_argument("--dest", default="assets")
    args = parser.parse_args()

    base_url, entry = ASSETS[args.asset]
    dest_dir = Path(args.dest) / base_url.rsplit("/", 1)[-1]

    print(f"fetching {args.asset} into {dest_dir}")
    count, missing = fetch_closure(base_url, entry, dest_dir)
    fetch(f"{base_url}/LICENSE", dest_dir / "LICENSE")

    print(f"\n{count} new file(s); entry layer: {dest_dir / entry}")
    if missing:
        print(f"{len(missing)} reference(s) could not be fetched:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
    print("\nValidate with: uv run graft asset validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
