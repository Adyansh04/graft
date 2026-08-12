"""Asset source interface.

Scanning a real object to a mesh is a later milestone. It arrives as another
implementation of `AssetSource` — everything downstream consumes a
`ResolvedAsset` and never learns where the USD came from.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ResolvedAsset:
    usd_path: Path
    target_prim_path: str
    class_name: str


@runtime_checkable
class AssetSource(Protocol):
    def resolve(self) -> ResolvedAsset:
        """Produce a local USD file and the prim path of the target object.

        Implementations may download, convert, or reconstruct — by the time
        this returns, `usd_path` must exist on disk.
        """
        ...
