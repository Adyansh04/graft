"""Config schema. Every model forbids unknown keys so a typo fails at load
rather than silently taking a default.

Several fields are `Literal` rather than plain values. Those are external
constraints, not preferences — changing one requires a code edit and a read
of the ADR that explains why. See local_docs/adr/0004.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Cosmos Transfer1 resolution buckets. Landscape is (1280, 704); anything
# else is silently resized and center-cropped, which invalidates every
# normalized label coordinate we carry through.
COSMOS_LANDSCAPE_RES = (1280, 704)

# Cosmos hardcodes 121 frames per clip. Fewer produces visible noise.
COSMOS_FRAMES_PER_CLIP = 121

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Positive = Annotated[int, Field(gt=0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunCfg(Strict):
    name: str
    seed: int = 0
    out_root: str = "runs"


class AssetCfg(Strict):
    usd_path: str
    target_prim_path: str
    # Sanity bounds for the asset's largest dimension, in metres. Catches
    # centimetre-authored assets that would otherwise render as a speck.
    expected_size_m: tuple[float, float] = (0.03, 0.40)

    @model_validator(mode="after")
    def _ordered(self):
        lo, hi = self.expected_size_m
        if lo >= hi:
            raise ValueError(f"expected_size_m must be (min, max) with min < max, got {self.expected_size_m}")
        return self


class ClassSpec(Strict):
    """One detectable class. Its id is its index in `Config.classes`.

    The colour is the segmentation colour handed to CosmosWriter's
    `segmentation_mapping`, so it must be distinguishable after the video
    round-trip. Labels are never read back from video, so this only needs to
    survive well enough for QA to decode.
    """

    name: str
    color: tuple[int, int, int, int]

    @model_validator(mode="after")
    def _channels_in_range(self):
        if not all(0 <= c <= 255 for c in self.color):
            raise ValueError(f"class {self.name!r}: colour channels must be 0-255, got {self.color}")
        return self


class SettleCfg(Strict):
    """Thresholds for deciding a dropped object has come to rest."""

    lin_vel_thresh: float = 0.1
    ang_vel_thresh: float = 0.1
    timeout_steps: Positive = 600


class SimCfg(Strict):
    resolution: tuple[Positive, Positive] = COSMOS_LANDSCAPE_RES
    rt_subframes: Positive = 6
    dlss_exec_mode: Literal[0, 1, 2, 3] = 2
    settle: SettleCfg = SettleCfg()

    @model_validator(mode="after")
    def _warn_off_bucket(self):
        if tuple(self.resolution) != COSMOS_LANDSCAPE_RES:
            raise ValueError(
                f"resolution {tuple(self.resolution)} is not the Cosmos landscape bucket "
                f"{COSMOS_LANDSCAPE_RES}. Cosmos silently resizes and center-crops anything "
                "else, which invalidates every label coordinate. Render natively at the "
                "bucket size."
            )
        return self


class CameraCfg(Strict):
    orbit_degrees: float = 120.0
    elevation_deg: tuple[float, float] = (15.0, 45.0)
    radius_m: tuple[float, float] = (0.4, 0.9)


class LightingCfg(Strict):
    dome_intensity: tuple[float, float] = (300.0, 1500.0)
    n_area_lights: tuple[int, int] = (1, 3)
    area_intensity: tuple[float, float] = (1e4, 8e4)
    # Fraction of area lights switched off per clip. Uneven, sometimes-dark
    # scenes teach more than uniformly lit ones.
    off_probability: Fraction = 0.3


class DistractorCfg(Strict):
    count: tuple[int, int] = (0, 4)
    pool_dir: str | None = None


class RandomizersCfg(Strict):
    lighting: LightingCfg = LightingCfg()
    distractors: DistractorCfg = DistractorCfg()


class CaptureCfg(Strict):
    n_clips: Positive = 8
    # Not configurable: Cosmos hardcodes 121. Present for readability only.
    frames_per_clip: Literal[COSMOS_FRAMES_PER_CLIP] = COSMOS_FRAMES_PER_CLIP
    negative_clip_fraction: Fraction = 0.0
    camera: CameraCfg = CameraCfg()
    randomizers: RandomizersCfg = RandomizersCfg()


class CosmosWeights(Strict):
    """Per-modality control weights.

    Defaults are Isaac Lab's validated combination, expressed in the modern
    JSON spec form. `vis` is the control that preserves the source's
    background and lighting, so it actively fights a lighting restyle — for
    heavy lighting change, drop vis/edge and raise depth/seg instead.
    """

    vis: Fraction = 0.3
    edge: Fraction = 0.3
    depth: Fraction = 0.6
    seg: Fraction = 0.7


class PromptCfg(Strict):
    sections_file: str = "configs/prompts.yaml"
    # The single non-random clause stating what must NOT change and what it
    # currently looks like. Cosmos guidance is explicit that this matters.
    invariant: str


class CosmosCfg(Strict):
    fps: Annotated[int, Field(ge=12, le=40)] = 30
    sigma_max: Annotated[float, Field(ge=0.0, le=90.0)] = 50.0
    # Set explicitly: transfer.py's argparse default is 5 while the docs say
    # 7, and argparse wins. Pinning it removes the ambiguity.
    guidance: float = 5.0
    seed: int = 1
    weights: CosmosWeights = CosmosWeights()
    prompt: PromptCfg

    @model_validator(mode="after")
    def _weight_ceiling(self):
        total = self.weights.vis + self.weights.edge + self.weights.depth + self.weights.seg
        if total > 2.0:
            raise ValueError(
                f"cosmos control weights sum to {total:.2f}; documented ceiling is 2.0"
            )
        return self


class EncodeCfg(Strict):
    """Our own ffmpeg encode from the writer's PNGs.

    crf 12 is well below visually lossless. crf 0 would buy nothing while
    4:2:0 chroma subsampling is in the chain, and costs several GB per clip.
    """

    crf: Annotated[int, Field(ge=0, le=51)] = 12
    preset: str = "slow"
    pix_fmt: str = "yuv420p"
    fps: Annotated[int, Field(ge=12, le=40)] = 30


class SplitCfg(Strict):
    train: Fraction = 0.8
    val: Fraction = 0.2
    # Not configurable. Consecutive frames within a clip are near-duplicates;
    # splitting by frame leaks them across train/val and inflates val scores.
    by: Literal["clip"] = "clip"

    @model_validator(mode="after")
    def _sums_to_one(self):
        total = self.train + self.val
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        return self


class DatasetCfg(Strict):
    sources: list[Literal["sim", "cosmos"]] = ["sim"]
    # Consecutive frames are near-duplicates; taking all of them inflates
    # dataset size without adding information.
    stride: Positive = 10
    split: SplitCfg = SplitCfg()
    formats: list[Literal["yolo_detect", "yolo_seg", "coco", "kitti"]] = ["yolo_detect"]

    @model_validator(mode="after")
    def _non_empty(self):
        if not self.sources:
            raise ValueError("dataset.sources must not be empty")
        if not self.formats:
            raise ValueError("dataset.formats must not be empty")
        return self


class CosmosQaCfg(Strict):
    # Object must survive the restyle...
    in_mask_ssim_min: Fraction = 0.7
    # ...while the background must actually change.
    out_mask_change_min: Fraction = 0.05


class QaCfg(Strict):
    blur_lap_var_min: float = 60.0
    min_bbox_area_px: Positive = 64
    cosmos: CosmosQaCfg = CosmosQaCfg()
    action: Literal["quarantine", "fail"] = "quarantine"


class TrainCfg(Strict):
    model: str = "yolo11n.pt"
    epochs: Positive = 10
    imgsz: Positive = 640
    batch: Positive = 8


class EvalCfg(Strict):
    """Evaluation targets.

    `real_photos_dir` is the one that counts. Sim-val shares the renderer,
    asset and randomization distribution with training, so a model can score
    well on it by learning Isaac's rendering characteristics. It is retained
    as a diagnostic — a large sim-to-real gap is itself the signal.
    """

    sim_val: bool = True
    real_photos_dir: str | None = None


class Config(Strict):
    run: RunCfg
    asset: AssetCfg
    classes: list[ClassSpec]
    sim: SimCfg = SimCfg()
    capture: CaptureCfg = CaptureCfg()
    cosmos: CosmosCfg
    encode: EncodeCfg = EncodeCfg()
    dataset: DatasetCfg = DatasetCfg()
    qa: QaCfg = QaCfg()
    train: TrainCfg = TrainCfg()
    eval: EvalCfg = EvalCfg()

    @model_validator(mode="after")
    def _classes_usable(self):
        if not self.classes:
            raise ValueError("at least one class is required")
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate class names: {names}")
        colors = [c.color for c in self.classes]
        if len(set(colors)) != len(colors):
            raise ValueError("duplicate class colours; segmentation decode would be ambiguous")
        return self

    def class_id(self, name: str) -> int:
        """Class id is the index in `classes` — the single ordering that the
        sim writer, the dataset `names:` list and QA all rely on."""
        for i, spec in enumerate(self.classes):
            if spec.name == name:
                return i
        raise KeyError(f"class {name!r} not in config.classes ({[c.name for c in self.classes]})")

    def segmentation_mapping(self) -> dict[str, list[int]]:
        """The mapping handed to CosmosWriter."""
        return {c.name: list(c.color) for c in self.classes}

    def class_names(self) -> list[str]:
        """Ordered names for a dataset yaml. Order is the class id order."""
        return [c.name for c in self.classes]
