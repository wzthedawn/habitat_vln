# 评估实验加速优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过禁用视频/轨迹图输出和GPU并行分配，实现评估实验8倍加速。

**Architecture:** 在EpisodeOutputManager添加开关参数控制视频/轨迹图生成；通过CUDA_VISIBLE_DEVICES环境变量分配GPU，并行运行4个实验。

**Tech Stack:** Python argparse, Bash脚本, CUDA环境变量

---

## 文件结构

| 文件 | 负责 |
|------|------|
| `utils/episode_output.py` | EpisodeOutputManager - 添加enable_video/enable_trajectory参数 |
| `run_vln_experiment.py:147,1171` | 传递禁用参数，条件调用视频生成 |
| `scripts/run_emergency_eval.py:577-593` | 添加--no-video/--no-trajectory命令行参数 |
| `scripts/run_parallel_eval.sh` | 新建：4GPU并行启动脚本 |

---

### Task 1: 修改EpisodeOutputManager添加开关参数

**Files:**
- Modify: `utils/episode_output.py:90-100`
- Modify: `utils/episode_output.py:327-365`

- [ ] **Step 1: 修改__init__方法添加参数**

```python
def __init__(
    self,
    output_dir: str = "results",
    enable_video: bool = True,
    enable_trajectory: bool = True,
):
    self.output_dir = Path(output_dir)
    self.enable_video = enable_video
    self.enable_trajectory = enable_trajectory
    self.logger = logging.getLogger("EpisodeOutputManager")
    self.current_episode: Optional[EpisodeOutput] = None

    # Create output directory
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.logger.info(f"Output directory: {self.output_dir.absolute()}")
    if not self.enable_video:
        self.logger.info("Video generation disabled")
    if not self.enable_trajectory:
        self.logger.info("Trajectory plot generation disabled")
```

- [ ] **Step 2: 修改finish_episode方法添加条件控制**

找到第354-356行（save_trajectory_plot调用），修改为条件调用：

```python
        # Save trajectory plot (if enabled)
        if goal_position and self.enable_trajectory:
            self.save_trajectory_plot(trajectory, goal_position, reference_path)
```

- [ ] **Step 3: 验证修改**

Run: `python -c "from utils.episode_output import EpisodeOutputManager; m = EpisodeOutputManager('test', enable_video=False, enable_trajectory=False); print('OK')"`
Expected: 输出 "OK"

- [ ] **Step 4: Commit**

```bash
git add utils/episode_output.py
git commit -m "feat: add enable_video/enable_trajectory parameters to EpisodeOutputManager"
```

---

### Task 2: 修改run_vln_experiment.py传递参数

**Files:**
- Modify: `run_vln_experiment.py:147` (EpisodeOutputManager instantiation)
- Modify: `run_vln_experiment.py:1171` (create_summary_video call)

- [ ] **Step 1: 修改MultiAgentVLNEvaluator.__init__添加参数接收**

在第147行找到EpisodeOutputManager创建，修改配置：

先在`__init__`方法开头添加参数接收（约第100行附近）：

```python
def __init__(self, config: Dict[str, Any]):
    """Initialize evaluator"""
    self.config = config
    self.logger = logging.getLogger("MultiAgentVLNEvaluator")

    # Video/Trajectory control
    self.enable_video = config.get("enable_video", True)
    self.enable_trajectory = config.get("enable_trajectory", True)
```

然后修改第147行：

```python
        self.output_manager = EpisodeOutputManager(
            config.get("output_dir", "results"),
            enable_video=self.enable_video,
            enable_trajectory=self.enable_trajectory,
        )
```

- [ ] **Step 2: 修改create_summary_video调用为条件调用**

在第1171行找到视频生成调用，修改为：

```python
                # Try to create summary video (if enabled)
                if self.enable_video:
                    self.output_manager.create_summary_video(fps=5)
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 4: Commit**

```bash
git add run_vln_experiment.py
git commit -m "feat: pass enable_video/enable_trajectory to EpisodeOutputManager"
```

---

### Task 3: 修改run_emergency_eval.py添加命令行参数

**Files:**
- Modify: `scripts/run_emergency_eval.py:577-607` (argparse section)

- [ ] **Step 1: 添加命令行参数**

在第591行后添加两个新参数：

```python
    parser.add_argument("--use-remote-llm", action="store_true", default=False)
    parser.add_argument("--llm-server", type=str, default="http://localhost:8000")
    parser.add_argument("--no-video", action="store_true", default=False,
                       help="Disable video generation for faster evaluation")
    parser.add_argument("--no-trajectory", action="store_true", default=False,
                       help="Disable trajectory plot generation")
```

- [ ] **Step 2: 将参数传递到config**

在第596-607行的config字典中添加：

```python
    config = {
        "mp3d_path": args.mp3d_path,
        "emergency_dataset_path": args.emergency_dataset,
        "max_steps": args.max_steps,
        "success_distance": 3.0,
        "device": "cuda",
        "use_int8": True,
        "use_remote_llm": args.use_remote_llm,
        "llm_server": args.llm_server,
        "sequence_length": 5,
        "adaptive_sequence": False,
        "enable_video": not args.no_video,
        "enable_trajectory": not args.no_trajectory,
    }
```

- [ ] **Step 3: 验证参数解析**

Run: `python scripts/run_emergency_eval.py --help | grep -E "no-video|no-trajectory"`
Expected: 显示两个新参数的帮助文本

- [ ] **Step 4: Commit**

```bash
git add scripts/run_emergency_eval.py
git commit -m "feat: add --no-video and --no-trajectory CLI arguments"
```

---

### Task 4: 创建并行启动脚本

**Files:**
- Create: `scripts/run_parallel_eval.sh`

- [ ] **Step 1: 创建脚本文件**

```bash
#!/bin/bash
# scripts/run_parallel_eval.sh
#
# Parallel evaluation script using 4 GPUs
# GPU allocation:
#   GPU 0: vLLM server + exp-c
#   GPU 1: baseline
#   GPU 2: exp-a
#   GPU 3: exp-b

set -e

# Configuration
EPISODES=${1:-10}
NO_VIDEO="--no-video --no-trajectory"
OUTPUT_DIR="results/emergency_eval_parallel"

echo "========================================"
echo "Parallel Emergency Evaluation"
echo "========================================"
echo "Episodes per experiment: $EPISODES"
echo "Video/Trajectory: DISABLED"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Create output directory
mkdir -p $OUTPUT_DIR

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Start 4 experiments in parallel
echo "Starting experiments..."

# GPU 1: baseline
CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp baseline --episodes $EPISODES --output "$OUTPUT_DIR/baseline" $NO_VIDEO \
    2>&1 | tee "$OUTPUT_DIR/baseline_log.txt" &
PID1=$!
echo "Started baseline on GPU 1 (PID: $PID1)"

# GPU 2: exp-a (PathReplanner only)
CUDA_VISIBLE_DEVICES=2 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp exp-a --episodes $EPISODES --output "$OUTPUT_DIR/exp-a" $NO_VIDEO \
    2>&1 | tee "$OUTPUT_DIR/exp-a_log.txt" &
PID2=$!
echo "Started exp-a on GPU 2 (PID: $PID2)"

# GPU 3: exp-b (PathReplanner + LoRA)
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp exp-b --episodes $EPISODES --output "$OUTPUT_DIR/exp-b" $NO_VIDEO \
    2>&1 | tee "$OUTPUT_DIR/exp-b_log.txt" &
PID3=$!
echo "Started exp-b on GPU 3 (PID: $PID3)"

# GPU 0: exp-c (LoRA only, shares with vLLM)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp exp-c --episodes $EPISODES --output "$OUTPUT_DIR/exp-c" $NO_VIDEO \
    2>&1 | tee "$OUTPUT_DIR/exp-c_log.txt" &
PID4=$!
echo "Started exp-c on GPU 0 (PID: $PID4)"

echo ""
echo "All 4 experiments running in parallel."
echo "Waiting for completion..."
echo ""

# Wait for all processes
wait $PID1
echo "baseline completed (exit code: $?)"

wait $PID2
echo "exp-a completed (exit code: $?)"

wait $PID3
echo "exp-b completed (exit code: $?)"

wait $PID4
echo "exp-c completed (exit code: $?)"

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON summary files found"
```

- [ ] **Step 2: 设置脚本权限**

Run: `chmod +x scripts/run_parallel_eval.sh`
Expected: 无错误输出

- [ ] **Step 3: 验证脚本语法**

Run: `bash -n scripts/run_parallel_eval.sh`
Expected: 无错误输出

- [ ] **Step 4: Commit**

```bash
git add scripts/run_parallel_eval.sh
git commit -m "feat: add parallel evaluation script for 4-GPU setup"
```

---

### Task 5: 集成测试

**Files:**
- None (testing existing changes)

- [ ] **Step 1: 测试单episode禁用视频**

Run: `python scripts/run_emergency_eval.py --exp baseline --episodes 1 --no-video --no-trajectory`
Expected: 完成无错误，results目录无episode_video.mp4和trajectory_plot.png

- [ ] **Step 2: 检查结果JSON正确保存**

Run: `ls results/episode*/agent_outputs.json | head -1`
Expected: 存在JSON文件

- [ ] **Step 3: 验证并行脚本启动（可选，需4GPU）**

Run: `./scripts/run_parallel_eval.sh 1` (仅1 episode测试)
Expected: 4进程并行启动并完成

---

## 验收清单

- [ ] EpisodeOutputManager接受enable_video/enable_trajectory参数
- [ ] run_vln_experiment.py正确传递参数
- [ ] run_emergency_eval.py接受--no-video/--no-trajectory参数
- [ ] 禁用视频时无episode_video.mp4生成
- [ ] 禁用轨迹时无trajectory_plot.png生成
- [ ] 并行脚本可正确启动4进程
- [ ] 所有commit已提交