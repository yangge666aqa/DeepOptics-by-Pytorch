# 项目整理与迁移

| 旧位置 | 新位置 |
| --- | --- |
| `Spectraltask.py` | `deepoptics/config.py` |
| `constant.py` | `deepoptics/constants.py` |
| `trainer.py` 中的训练逻辑 | `deepoptics/train.py` |
| `evaluate.py` | `deepoptics/evaluate.py` |
| `loss.py` / `metrics.py` | `deepoptics/losses.py` / `deepoptics/metrics.py` |
| `optics/` | `deepoptics/optics/` |
| `net/` | `deepoptics/models/` |
| `util/data/` | `deepoptics/data/` |
| `util/pytorch_ssim.py` | `deepoptics/utils/pytorch_ssim.py` |
| `MTF.gif` | `docs/assets/MTF.gif` |

`python trainer.py` 仍可启动训练，`main.py` 的 PyCharm 示例被替换为同一训练入口。外部脚本中的旧模块导入需要按上表修改。

配置模块不再创建全局 SummaryWriter 或输出目录。单独使用 `Camera` 时，可显式传入 `writer` 和 `output_dir`；默认不输出诊断图。自定义实例还可传入 `doe_args`、`propagation_args`、`network_args`；这些配置的空间尺寸和波段数需要一致。模型在目标设备构造并 `.to(device)` 后调用 `.done()`，再执行前向计算。

本次保留原始光学传播、连续高度优化和网络结构。验证切换为评估模式，因此 BatchNorm 不再更新验证统计量；这可能导致指标与旧训练脚本不同。数据读取统一返回 float32，关闭 HDF5 句柄，并按 worker 分配文件，避免多 worker 重复读取。

IDE 文件、缓存、数据和训练输出不再提交到 Git；本机已有文件仍保留。原先的空 `tensorboard` 文件已移除，使用 `tensorboard --logdir runs` 启动日志界面。

原有模块类名、模型参数名尽量保留。旧 `state_dict` 可通过新相同配置的模型加载；完整 pickle 模型因模块路径改变不保证兼容。当前检查点仍只保存权重，不自动恢复优化器、学习率或步数。
