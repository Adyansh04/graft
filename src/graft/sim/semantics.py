"""Semantic labelling of the target, and removal of everything else.

Library assets ship their own labels — the SimReady mug carries a
`wikidata_class` taxonomy, and warehouse props come pre-tagged. Any of those
left in place would appear in annotations as spurious classes, so every
label that is not ours is stripped.

Distractors are never labelled: they must occlude and clutter without ever
producing an annotation.
"""

TAXONOMY = "class"

# Instances can nest; this bounds the re-walk rather than looping forever.
MAX_INSTANCE_DEPTH = 8


def make_editable(prim_path: str) -> list[str]:
    """Clear the instanceable flag on a subtree.

    SimReady assets ship their geometry inside instanceable prims, and USD
    refuses to author onto an instance proxy:

        Cannot create prim spec at <...>; authoring to an instance proxy is
        not allowed.

    Clearing the flag turns the proxies into real prims that can carry
    labels. Returns the paths that were changed.
    """
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"cannot edit missing prim {prim_path}")

    cleared: list[str] = []
    # Clearing an outer flag expires the prim handles beneath it and can
    # expose nested instances, so re-walk and re-fetch by path each pass.
    for _ in range(MAX_INSTANCE_DEPTH):
        root = stage.GetPrimAtPath(prim_path)
        paths = [
            str(p.GetPath())
            for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if p.IsInstance()
        ]
        if not paths:
            break
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid() and prim.IsInstance():
                prim.SetInstanceable(False)
                cleared.append(path)
    return cleared


def label_target(prim_path: str, class_name: str) -> list[str]:
    """Label the target subtree.

    Labels go on the meshes themselves, not only the reference root — that
    is what the annotators actually resolve. Returns the labelled paths.
    """
    import omni.usd
    from isaacsim.core.experimental.utils.semantics import add_labels
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"cannot label missing prim {prim_path}")

    make_editable(prim_path)

    labelled = []
    add_labels(prim, labels=[class_name], taxonomy=TAXONOMY)
    labelled.append(str(prim.GetPath()))
    for mesh in Usd.PrimRange(prim):
        if mesh.GetTypeName() == "Mesh":
            add_labels(mesh, labels=[class_name], taxonomy=TAXONOMY)
            labelled.append(str(mesh.GetPath()))

    if len(labelled) == 1:
        raise RuntimeError(
            f"{prim_path} contains no Mesh prims to label — annotations would be empty"
        )
    return labelled


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
