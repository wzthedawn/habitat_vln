---
name: 评估实验加速优化设计
description: 通过禁用视频输出和GPU并行分配实现评估实验加速
type: project
---

# 评估实验加速优化设计

## 背景

当前评估实验运行过慢（每个episode约2分钟，10个episode×4实验=约80分钟），需要加速以支持快速迭代验证。

## 目标

在保证评估指标质量的前提下，实现以下优化：
1. 测试算法是否有效
2. 知道当前导航准确率
3. 验证微调模型是否有用

## 现有资源

- 4张24GB显卡
- vLLM服务器运行LLM推理
- 应急评估脚本 `scripts/run_emergency_eval.py`

## 优化方案

### 1. 禁用视频和轨迹图输出

**Why:** 视频和轨迹图生成是纯可视化功能，不影响SR/SPL等评估指标。当前每个episode生成视频约消耗50%时间。

**实现:**
- 修改 `utils/episode_output.py`，添加 `enable_video` 和 `enable_trajectory` 参数
- 修改 `run_vln_experiment.py`，传递禁用配置
- 修改 `scripts/run_emergency_eval.py`，添加 `--no-video` 和 `--no-trajectory` 命令行参数

### 2. 4GPU并行运行4实验

**Why:** 4张独立显卡可完全隔离资源，无并发风险。

**GPU分配方案:**
- GPU 0: vLLM服务器（已运行）
- GPU 1: baseline实验
- GPU 2: exp-a实验
- GPU 3: exp-b实验
- GPU 0（共享）: exp-c实验（与vLLM同GPU，Habitat渲染压力较小）

**How to apply:** 使用 `CUDA_VISIBLE_DEVICES` 控制每个进程的GPU访问。

## 实现细节

### 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `utils/episode_output.py` | 添加 enable_video/enable_trajectory 参数 |
| `run_vln_experiment.py` | 传递视频禁用配置 |
| `scripts/run_emergency_eval.py` | 添加 --no-video/--no-trajectory 参数 |
| `scripts/run_parallel_eval.sh` | 新建：4GPU并行启动脚本 |

### EpisodeOutputManager 修改

```python
class EpisodeOutputManager:
    def __init__(
        self,
        output_dir: str,
        enable_video: bool = True,
        enable_trajectory: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.enable_video = enable_video
        self.enable_trajectory = enable_trajectory

    def finish_episode(self, episode_id, success, ...):
        # 保存agent_outputs.json（必须）
        self.save_agent_outputs(...)

        # 轨迹图（可选）
        if self.enable_trajectory:
            self.save_trajectory_plot(...)

        # 视频（可选）
        if self.enable_video:
            self.create_summary_video(fps=5)
```

### 并行启动脚本

```bash
#!/bin/bash
# scripts/run_parallel_eval.sh

# 禁用视频和轨迹图
NO_VIDEO="--no-video --no-trajectory"

# GPU分配
# GPU 0: vLLM已在运行，exp-c 共享此GPU
# GPU 1-3: 独立运行其他实验

echo "Starting parallel evaluation experiments..."

# 启动4个实验，各使用独立GPU
CUDA_VISIBLE_DEVICES=1 python scripts/run_emergency_eval.py --exp baseline --episodes 10 $NO_VIDEO &
CUDA_VISIBLE_DEVICES=2 python scripts/run_emergency_eval.py --exp exp-a --episodes 10 $NO_VIDEO &
CUDA_VISIBLE_DEVICES=3 python scripts/run_emergency_eval.py --exp exp-b --episodes 10 $NO_VIDEO &
CUDA_VISIBLE_DEVICES=0 python scripts/run_emergency_eval.py --exp exp-c --episodes 10 $NO_VIDEO &

echo "All experiments started. Waiting for completion..."
wait

echo "All experiments completed. Results saved to results/emergency_eval/"
```

## 预期效果

| 项目 | 原耗时 | 优化后 |
|------|--------|--------|
| 单episode | ~2分钟 | ~1分钟（禁用视频） |
| 10 episodes串行 | ~20分钟 | ~10分钟 |
| 4实验并行（4GPU） | ~80分钟 | ~10分钟 |

**总加速：约8倍**

## 验证方法

1. 先单独运行一个episode验证禁用视频功能正常
2. 检查评估结果JSON文件确保指标正确保存
3. 运行完整并行脚本，观察4进程是否正常启动
4. 对比优化前后评估结果确保一致性

## 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| GPU资源不足 | 低 | 每张24GB足够，独立分配 |
| vLLM并发请求排队 | 低 | vLLM天然支持并发 |
| 评估指标变化 | 无 | 视频不影响指标 |