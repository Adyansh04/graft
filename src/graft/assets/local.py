"""A USD file already on disk."""

from pathlib import Path

from graft.assets.base import ResolvedAsset


class LocalUSDSource:
    def __init__(self, usd_path: str | Path, target_prim_path: str, class_name: str):
        self.usd_path = Path(usd_path)
        self.target_prim_path = target_prim_path
        self.class_name = class_name

    def resolve(self) -> ResolvedAsset:
        if not self.usd_path.is_file():
            raise FileNotFoundError(
                f"asset not found: {self.usd_path}. Assets live outside git — see "
                "scripts/fetch_asset.sh."
            )
        return ResolvedAsset(
            usd_path=self.usd_path,
            target_prim_path=self.target_prim_path,
            class_name=self.class_name,
        )
