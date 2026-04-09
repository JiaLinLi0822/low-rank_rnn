"""
完整的 tasks_list 示例
包含所有可用的任务类型，用于 continual learning 训练
"""

from data.functions import (
    # 基础 1D 函数
    X, X2, X3, X4, X5, X6,
    # 旋转和反转变体
    X2Rotate, X2RotateReverse, X2Reverse,
    X3Rotate, X3RotateReverse,
    X4Rotate, X4RotateReverse, X4RotateMinus,
    X5Rotate, X5RotateReverse,
    X6Rotate, X6RotateReverse,
    XReverse,
    # 三角函数
    Sine, CoSine, Tanh, Tan,
    # 其他 1D 函数
    Triangle, Poly2, Step1D, Bump1D,
    # 2D 函数
    L1, L2, L3, L4, L6, LMAX,
    L1Reverse, L190, L1270,
    L2180, L2180Reverse, L290, L2270,
    L3180, L3180Reverse,
    L4180, L4180Reverse, L490, L4270,
    L6180, L6180Reverse, L690, L6270,
    LMAX180, LMAX90, LMAX1270,
    # 其他
    Square, SquareReverse, Identity
)

from data.custom_data_generator import (
    # Gated 版本 (标准)
    FlipFlopGenerator,
    CyclesGenerator,
    LinesGenerator,
    FlipFlopCycleGenerator,
    FlipFlopLineGenerator,
    LimitCycleLineGenerator,
    # Parallel 版本
    ParallelFlipFlopGenerator,
    ParallelCyclesGenerator,
    ParallelLineGenerator,
    ParallelFlipFlopCycleGenerator,
    ParallelCycleLineGenerator,
    ParallelFlipFlopLineGenerator,
    # Orthogonal 版本
    OrthogonalFlipFlopGenerator,
    OrthogonalLineGenerator,
    OrthogonalCyclesGenerator,
    OrthogonalFlipFlopCycleGenerator,
    OrthogonalCycleLineGenerator,
    OrthogonalFlipFlopLineGenerator,
    # 其他
    SineWaveGenerator,
    RingGenerator
)

# ============================================
# 示例 1: 渐进式函数任务 (推荐用于开始)
# ============================================
tasks_list_function_progressive = [
    [X()],                                    # Stage 1: 基础线性函数
    [X(), X2()],                              # Stage 2: 添加二次函数
    [X(), X2(), X2Rotate()],                 # Stage 3: 添加旋转变体
    [X(), X2(), X2Rotate(), XReverse()],    # Stage 4: 添加反转
    [X(), X2(), X2Rotate(), XReverse(), X4()],  # Stage 5: 添加四次函数
    [X(), X2(), X2Rotate(), XReverse(), X4(), X4Rotate()],  # Stage 6: 添加四次旋转
]

# ============================================
# 示例 2: 完整的多项式函数系列
# ============================================
tasks_list_polynomials = [
    [X()],
    [X(), X2()],
    [X(), X2(), X3()],
    [X(), X2(), X3(), X4()],
    [X(), X2(), X3(), X4(), X5()],
    [X(), X2(), X3(), X4(), X5(), X6()],
]

# ============================================
# 示例 3: 混合函数类型 (多项式 + 三角函数)
# ============================================
tasks_list_mixed_functions = [
    [X()],
    [X(), X2()],
    [X(), X2(), Sine()],
    [X(), X2(), Sine(), CoSine()],
    [X(), X2(), Sine(), CoSine(), Tanh()],
    [X(), X2(), Sine(), CoSine(), Tanh(), X4()],
]

# ============================================
# 示例 4: 2D 函数任务
# ============================================
tasks_list_2d_functions = [
    [L1()],
    [L1(), L2()],
    [L1(), L2(), L3()],
    [L1(), L2(), L3(), L4()],
    [L1(), L2(), L3(), L4(), LMAX()],
]

# ============================================
# 示例 5: Flip-Flop 任务 (渐进式)
# ============================================
tasks_list_flipflop_progressive = [
    [FlipFlopGenerator(n_bits=2)],                    # Stage 1: 2-bit flip-flop
    [FlipFlopGenerator(n_bits=2), 
     FlipFlopGenerator(n_bits=3)],                    # Stage 2: 添加 3-bit
    [FlipFlopGenerator(n_bits=2), 
     FlipFlopGenerator(n_bits=3),
     ParallelFlipFlopGenerator(n_bits=2)],           # Stage 3: 添加 parallel
    [FlipFlopGenerator(n_bits=2), 
     FlipFlopGenerator(n_bits=3),
     ParallelFlipFlopGenerator(n_bits=2),
     OrthogonalFlipFlopGenerator(n_bits=2)],         # Stage 4: 添加 orthogonal
]

# ============================================
# 示例 6: 不同 Flip-Flop 模式 (相同位数)
# ============================================
tasks_list_flipflop_modes = [
    [FlipFlopGenerator(n_bits=2)],                    # Stage 1: Gated
    [FlipFlopGenerator(n_bits=2),
     ParallelFlipFlopGenerator(n_bits=2)],           # Stage 2: 添加 Parallel
    [FlipFlopGenerator(n_bits=2),
     ParallelFlipFlopGenerator(n_bits=2),
     OrthogonalFlipFlopGenerator(n_bits=2)],        # Stage 3: 添加 Orthogonal
]

# ============================================
# 示例 7: 混合任务 (函数 + 记忆任务)
# ============================================
tasks_list_mixed_tasks = [
    [X()],                                            # Stage 1: 简单函数
    [X(), X2()],                                      # Stage 2: 添加函数
    [X(), X2(), FlipFlopGenerator(n_bits=2)],        # Stage 3: 添加记忆任务
    [X(), X2(), FlipFlopGenerator(n_bits=2),
     ParallelFlipFlopGenerator(n_bits=2)],           # Stage 4: 添加并行记忆
]

# ============================================
# 示例 8: 完整的 6 任务序列 (用于 Rank2 架构)
# ============================================
tasks_list_6tasks_rank2 = [
    [X()],
    [X(), X2()],
    [X(), X2(), X2Rotate()],
    [X(), X2(), X2Rotate(), XReverse()],
    [X(), X2(), X2Rotate(), XReverse(), X4()],
    [X(), X2(), X2Rotate(), XReverse(), X4(), X4Rotate()],
]

# ============================================
# 示例 9: 所有旋转和反转变体
# ============================================
tasks_list_all_variants = [
    [X()],
    [X(), X2()],
    [X(), X2(), X2Rotate()],
    [X(), X2(), X2Rotate(), X2RotateReverse()],
    [X(), X2(), X2Rotate(), X2RotateReverse(), XReverse()],
    [X(), X2(), X2Rotate(), X2RotateReverse(), XReverse(), X4()],
    [X(), X2(), X2Rotate(), X2RotateReverse(), XReverse(), X4(), X4Rotate()],
    [X(), X2(), X2Rotate(), X2RotateReverse(), XReverse(), X4(), X4Rotate(), X4RotateReverse()],
]

# ============================================
# 示例 10: 记忆任务完整系列
# ============================================
tasks_list_memory_complete = [
    [FlipFlopGenerator(n_bits=2)],
    [FlipFlopGenerator(n_bits=2), CyclesGenerator(n_bits=2)],
    [FlipFlopGenerator(n_bits=2), CyclesGenerator(n_bits=2), LinesGenerator(n_bits=2)],
    [FlipFlopGenerator(n_bits=2), CyclesGenerator(n_bits=2), LinesGenerator(n_bits=2),
     ParallelFlipFlopGenerator(n_bits=2)],
    [FlipFlopGenerator(n_bits=2), CyclesGenerator(n_bits=2), LinesGenerator(n_bits=2),
     ParallelFlipFlopGenerator(n_bits=2), OrthogonalFlipFlopGenerator(n_bits=2)],
]

# ============================================
# 使用示例
# ============================================
if __name__ == '__main__':
    from train_continual import ContinualLearningTrainer
    from model.pt_models import VanillaArchitecture, Rank2Architechture
    from model.model_wrapper import OptimizationParameters
    
    # 选择一个 tasks_list
    stages = tasks_list_function_progressive  # 或选择其他示例
    
    trainer = ContinualLearningTrainer(
        architecture_func=VanillaArchitecture,  # 或 Rank2Architechture
        units=100,
        tasks_list=stages,
        instance_range=range(0, 5),
        optimization_params=OptimizationParameters(
            batch_size=32,
            epochs=100,
            minimal_loss=1e-4,
            initial_lr=1e-4
        ),
        device='auto',
        recurrent_bias=False,
        readout_bias=True
    )
    
    # 训练所有阶段
    trainer.train_all_stages()
    
    print(f"\n训练完成！共 {len(stages)} 个阶段")
    print(f"每个阶段的任务数: {[len(stage) for stage in stages]}")

