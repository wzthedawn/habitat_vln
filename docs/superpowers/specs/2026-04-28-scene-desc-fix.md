---
title: DecisionAgent信息传递修复（nav_hint替换为scene_description）
date: 2026-04-28
type: design
status: completed
---

# DecisionAgent信息传递修复设计文档（最终版本）

## 修改概述

根据用户要求：
1. **删除nav_hint** - 移除所有nav_hint相关代码
2. **传入scene_description** - 将完整的场景描述传递给DecisionAgent
3. **增加max_token** - PerceptionAgent的max_new_tokens从250增加到400

## 修改文件清单

| 文件 | 改动内容 | 状态 |
|------|----------|------|
| `agents/perception_agent.py` | 删除nav_hint字段，增加max_new_tokens到400 | ✅ |
| `agents/decision_agent.py` | 删除nav_hint，添加scene_description到三种模式prompt | ✅ |
| `strategies/cot.py` | 删除nav_hint，改为scene_description分析 | ✅ |
| `run_vln_experiment.py` | 日志改为scene_description | ✅ |

## Prompt新结构（scene_description替代nav_hint）

### Easy模式
```
## 场景描述
Spacious living room with a large sofa on the left, TV on the wall ahead,
and a door on the right leading to the hallway. Open space in the center.

## 规则
2. 根据场景描述判断可行走方向
```

### Medium模式
```
## 场景描述
{scene_description}

## 规则
2. 根据场景描述判断可行走方向
```

### Hard模式
```
## 场景描述
{scene_description}

## 规则
2. 根据场景描述和观点汇总决策
```

## PerceptionAgent Prompt修改

**修改前**（要求生成nav_hint）：
```
"scene_brief":"short description in 10-15 words",
"nav_hint":"navigation hint in 20-30 words"
```

**修改后**（只要求详细场景描述）：
```
"scene_description":"detailed description in 30-50 words"
```

**场景描述要求**：
- 房间布局和大小
- 可见家具和物体
- 开放路径和门
- 楼梯（如有）

## max_new_tokens增加

| 位置 | 修改前 | 修改后 |
|------|--------|--------|
| PerceptionAgent generate_vision_dual | 250 | 400 |
| PerceptionAgent generate_vision | 250 | 400 |

## CoT Strategy修改

**修改前**：
```
Step 2 - Quote nav_hint: Find the sentence in nav_hint that relates to your goal keyword
```

**修改后**：
```
Step 2 - Scene analysis: Analyze scene description for navigation cues
- Identify key features: doors, corridors, open spaces, stairs
```

---

*设计文档版本：2026-04-28-v3*
*状态：已完成*