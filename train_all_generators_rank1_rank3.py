import shutil
from pathlib import Path

from data.custom_data_generator import (
    FlipFlopGenerator,
    CyclesGenerator,
    LinesGenerator,
    FlipFlopCycleGenerator,
    FlipFlopLineGenerator,
    LimitCycleLineGenerator,
    ParallelLineGenerator,
    ParallelFlipFlopGenerator,
    ParallelCyclesGenerator,
    ParallelFlipFlopCycleGenerator,
    ParallelCycleLineGenerator,
    ParallelFlipFlopLineGenerator,
    OrthogonalFlipFlopGenerator,
    OrthogonalLineGenerator,
    OrthogonalCyclesGenerator,
    OrthogonalFlipFlopCycleGenerator,
    OrthogonalCycleLineGenerator,
    OrthogonalFlipFlopLineGenerator,
)
from model.model_wrapper import InnerModelWrapper, ModelWrapper, OptimizationParameters
from model.pt_models import Rank1Architechture, Rank2Architechture, Rank3Architechture


# List of generator constructors (multi-task regime of two bits/tasks)
GENS = [
    ("flipflop", lambda: FlipFlopGenerator(n_bits=2)),
    ("cycles", lambda: CyclesGenerator(n_bits=2)),
    ("lines", lambda: LinesGenerator(n_bits=2)),
    # ("flipflopcycle", lambda: FlipFlopCycleGenerator(n_bits=2)),
    # ("flipflopline", lambda: FlipFlopLineGenerator(n_bits=2)),
    # ("limitcycleline", lambda: LimitCycleLineGenerator(n_bits=2)),
    # ("parallel_lines", lambda: ParallelLineGenerator(n_bits=2)),
    # ("parallel_flipflop", lambda: ParallelFlipFlopGenerator(n_bits=2)),
    # ("parallel_cycles", lambda: ParallelCyclesGenerator(n_bits=2)),
    # ("parallel_flipflopcycle", lambda: ParallelFlipFlopCycleGenerator(n_bits=2)),
    # ("parallel_cycleline", lambda: ParallelCycleLineGenerator(n_bits=2)),
    # ("parallel_flipflopline", lambda: ParallelFlipFlopLineGenerator(n_bits=2)),
    # ("orthogonal_flipflop", lambda: OrthogonalFlipFlopGenerator(n_bits=2)),
    # ("orthogonal_lines", lambda: OrthogonalLineGenerator(n_bits=2)),
    # ("orthogonal_cycles", lambda: OrthogonalCyclesGenerator(n_bits=2)),
    # ("orthogonal_flipflopcycle", lambda: OrthogonalFlipFlopCycleGenerator(n_bits=2)),
    # ("orthogonal_cycleline", lambda: OrthogonalCycleLineGenerator(n_bits=2)),
    # ("orthogonal_flipflopline", lambda: OrthogonalFlipFlopLineGenerator(n_bits=2)),
]

ARCHS = [
    ("rank2", Rank2Architechture),
]

opt = OptimizationParameters(epochs=100)
instance_range = range(100, 101)
units = 100


def copy_checkpoints(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    # initial weights
    init_file = src_dir / "initial_weights.pt"
    if init_file.exists():
        shutil.copy(init_file, dst_dir / "initial_weights.pt")
    # best/final weights
    final_file = src_dir / "weights.pt"
    if final_file.exists():
        shutil.copy(final_file, dst_dir / "final_weights.pt")
    # intermediate checkpoints weights{N}.pt
    for wfile in src_dir.glob("weights[0-9]*.pt"):
        shutil.copy(wfile, dst_dir / wfile.name)


def train_and_collect(gen_name: str, gen_fn, arch_label: str, arch_cls):
    gen = gen_fn()
    wrapper = ModelWrapper(
        arch_cls,
        units=units,
        train_data=gen,
        optimization_params=opt,
        instance_range=instance_range,
    )
    wrapper.train_model()

    for inst in instance_range:
        inner = InnerModelWrapper(wrapper.architecture, wrapper.name, inst)
        src_dir = Path(inner.model_path)
        dst_dir = Path(f"{arch_label}_wts") / gen_name / f"i{inst}"
        copy_checkpoints(src_dir, dst_dir)


if __name__ == "__main__":
    for gen_name, gen_fn in GENS:
        for arch_label, arch_cls in ARCHS:
            train_and_collect(gen_name, gen_fn, arch_label, arch_cls)
