---
name: Sequence Length Experiment Comparison
description: sequence-length=10与sequence-length=5的实验对比结果
type: project
---

## 实验对比（2026-04-26）

### 配置差异
- 之前实验: sequence-length=5
- 本次实验: sequence-length=10

### 结果对比

| 指标 | 5步序列 | 10步序列 | 变化 |
|------|---------|----------|------|
| 总耗时 | 1120秒 | 607.9秒 | **-45%** |
| Episode | 1 | 1 | 相同 |
| Max Steps | 150 | 150 | 相同 |
| 实际步数 | - | 130 | - |
| SR | - | 0.0% | - |
| NE | - | 3.43m | - |
| nDTW | - | 0.558 | - |

### 性能分析
- 耗时减少45%主要归因于减少LLM调用次数
- sequence-length=10使得每次LLM调用生成10个动作，减少了调用频率
- 虽然任务失败(SR=0%)，但性能指标验证了sequence-length参数的效果

### 实验命令
```bash
python run_vln_experiment.py \
    --use-remote-llm \
    --llm-server http://localhost:8000 \
    --model-path "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B" \
    --episodes 1 \
    --max-steps 150 \
    --sequence-length 10
```

**Why:** 验证更长序列是否能减少LLM调用开销，提高效率
**How to apply:** 后续实验可根据任务复杂度选择合适的sequence-length值