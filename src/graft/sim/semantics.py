"""Semantic labelling of the target, and removal of everything else.

Library assets ship their own labels — the SimReady mug carries a
`wikidata_class` taxonomy, and warehouse props come pre-tagged. Any of those
left in place would appear in annotations as spurious classes, so every
label that is not ours is stripped.

Distractors are never labelled: they must occlude and clutter without ever
producing an annotation.
"""

TAXONOMY = "class"


def label_target(prim_path: str, class_name: str) -> None:
    import omni.usd
    from isaacsim.core.experimental.utils.semantics import add_labels

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"cannot label missing prim {prim_path}")
    add_labels(prim, labels=[class_name], taxonomy=TAXONOMY)


def strip_foreign_labels(keep_prim_path: str) -> int:
    """Remove semantic labels from every prim outside the target subtree.

    Returns how many prims were stripped.
    """
    import omni.usd
    from isaacsim.core.experimental.utils.semantics import get_labels, remove_all_labels
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    keep = stage.GetPrimAtPath(keep_prim_path)
    keep_paths = (
        {str(p.GetPath()) for p in Usd.PrimRange(keep, Usd.TraverseInstanceProxies())}
        if keep and keep.IsValid()
        else set()
    )

    stripped = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if str(prim.GetPath()) in keep_paths:
            continue
        try:
            if not get_labels(prim):
                continue
            # remove_labels requires naming the labels; this drops all of
            # them, which is the point.
            remove_all_labels(prim)
            stripped += 1
        except Exception:  # noqa: BLE001 - prim may not support labels at all
            continue
    return stripped


def read_labels(prim_path: str) -> dict:
    import omni.usd
    from isaacsim.core.experimental.utils.semantics import get_labels

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return {}
    return get_labels(prim) or {}
