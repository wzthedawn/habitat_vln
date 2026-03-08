# VLN 评估指标分析报告

## 一、MSNav 论文实验结果

### 1.1 R2R-ZS 数据集 (72 scenes, 216 samples)

| 方法 | SR | SPL | OSR | NE |
|------|-----|-----|-----|-----|
| NavGPT | 34% | 29% | 42% | 6.46m |
| DiscussNav | 43% | 40% | 61% | 5.32m |
| MapGPT | 45.8% | 37.6% | 56.5% | 5.31m |
| **MSNav** | **50.9%** | **42.6%** | **63.9%** | **5.02m** |

### 1.2 R2R Val-Unseen 数据集 (11 scenes, 783 samples)

| 设置 | 方法 | SR | SPL | OSR | NE |
|------|------|-----|-----|-----|-----|
| ZS | NavGPT | 36.1% | 31.6% | 40.3% | 6.26m |
| ZS | MapGPT | 45.8% | 37.6% | 56.5% | 5.31m |
| **ZS** | **MSNav** | **46%** | **40%** | **65%** | **5.24m** |
| Train | EnvDrop | 52% | 48% | - | 5.22m |
| Pretrain | HAMT | 66% | 61% | 73% | 2.29m |
| Pretrain | ScaleVLN | 81% | 70% | 88% | 2.09m |

---

## 二、指标定义（微软VLN论文标准）

### 2.1 Success Rate (SR)
```
SR = 成功Episodes数 / 总Episodes数
成功 = 最终位置距目标 < 3米
```

### 2.2 SPL (Success weighted by Path Length)
```
SPL = (1/N) × Σ (success_i × l_i / max(p_i, l_i))

其中:
- success_i = 1 if 成功 else 0
- l_i = 最短路径长度
- p_i = 实际路径长度
```
**关键点**: SPL只有成功时才非零，且路径越高效SPL越高

### 2.3 Oracle Success Rate (OSR)
```
OSR = 轨迹上存在成功点的Episodes数 / 总Episodes数
```
表示如果能够选择最优停止点，能达到的最大成功率

### 2.4 Navigation Error (NE)
```
NE = (1/N) × Σ distance(final_position_i, goal_i)
```
平均最终距离误差，越小越好

---

## 三、当前实验结果分析

### 3.1 实验结果

| 指标 | 当前值 | MSNav ZS | 差距 | 原因 |
|------|--------|----------|------|------|
| SR | 10% | 50.9% | -40.9% | 随机策略无导航能力 |
| SPL | 0.079 | 0.426 | -0.347 | SR太低导致 |
| OSR | 15% | 63.9% | -48.9% | 无法有效接近目标 |
| NE | 5.22m | 5.02m | +0.2m | 相对合理 |

### 3.2 SPL 低的根本原因

**数学分析：**
```
假设:
- 总Episodes = 20
- 成功Episodes = 2 (SR = 10%)
- 成功Episode平均SPL = 0.79

则总SPL = (2 × 0.79) / 20 = 0.079
```

**结论：SPL低的主要原因是SR低**

### 3.3 改进方向

1. **提升SR** → 需要有效的导航策略
2. **提升路径效率** → 成功时路径要短
3. **使用LLM决策** → 理解指令并规划

---

## 四、MSNav 关键技术

### 4.1 Memory Module
- 动态拓扑地图
- 节点剪枝（解决LLM上下文限制）
- ME (Map Efficiency) 指标: 40.4%

### 4.2 Spatial Module
- Qwen-Sp: 空间推理模型
- I-O-S 数据集: 28,414 samples
- F1: 0.316, NDCG: 0.388

### 4.3 Decision Module
- GPT-4o 路径规划
- Prompt设计优化
- 集成地图和空间布局

---

## 五、性能对比总结

### Zero-Shot VLN 方法排名（按SR）

| 排名 | 方法 | SR | SPL |
|------|------|-----|-----|
| 1 | MSNav | 50.9% | 42.6% |
| 2 | MapGPT | 45.8% | 37.6% |
| 3 | DiscussNav | 43% | 40% |
| 4 | NavGPT | 34% | 29% |
| - | **当前(随机)** | **10%** | **7.9%** |

### 训练方法 vs Zero-Shot

| 类型 | 最佳方法 | SR | SPL | 差距 |
|------|---------|-----|-----|------|
| Pretrain | ScaleVLN | 81% | 70% | - |
| Train | EnvDrop | 52% | 48% | - |
| ZS | MSNav | 50.9% | 42.6% | - |
| **随机** | - | **10%** | **7.9%** | **SR差40%** |

---

## 六、结论

1. **当前SPL = 0.079 正确**：因为SR只有10%
2. **论文SPL = 42.6%**：因为SR达到50.9%
3. **要提升SPL**：必须先提升SR
4. **MSNav的优势**：
   - Memory Module: 长距离任务SR提升20%
   - Spatial Module: 终点识别准确
   - Decision Module: 路径规划高效

---

## 七、后续工作

1. 实现LLM导航策略替代随机策略
2. 添加Memory Module构建拓扑地图
3. 训练Spatial Module理解空间布局
4. 评估时增加nDTW和ME指标