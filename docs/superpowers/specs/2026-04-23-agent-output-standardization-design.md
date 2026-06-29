---
name: Agent输出格式统一与日志改进
description: 统一objects字段格式，改进日志输出，解决数据格式不一致和调试困难问题
type: project
---

# Agent输出格式统一与日志改进设计

## 背景

### 问题

实验过程中出现：
1. **数据格式不一致** - objects字段内部有时用"object"，有时用"name"，需要fallback逻辑处理
2. **不知道agent输出内容** - 日志分散在各agent内部，调试时难以查看完整输出
3. **实验失败问题定位困难** - 错误日志淹没在大量日志中，难以快速定位失败点

### 证据

```python
# decision_agent.py 需要fallback处理两种字段名
objects = [o.get("object", o.get("name", str(o))) for o in objects_raw[:3]]

# decision_agent.py 需要处理两种类型（字符串 vs Dict）
visible_objects = [{"name": o} if isinstance(o, str) else o for o in (objects or [])]
```

### 根因

- 各模块输出objects时字段名不统一
- 日志分散，缺少统一的输出摘要
- 错误日志不够醒目

## 解决方案

### 方案：统一格式 + 改进日志

**不引入TypedDict**（无法解决内部字段名问题），直接：
- 统一objects输出格式（只用"name"字段）
- 删除decision_agent中的fallback逻辑
- 添加汇总日志和错误高亮

### 改动范围

| 文件 | 改动 |
|------|------|
| models/model_manager.py | objects输出统一用"name"字段 |
| agents/perception_agent.py | objects输出统一用"name"字段 |
| agents/decision_agent.py | 删除fallback，直接读"name" |
| run_vln_experiment.py | 改进汇总日志 + 错误高亮 |

## 实现细节

### 1. 统一objects格式

**规定：objects始终是 `List[Dict]`，每个dict只用"name"字段**

```python
# 输出格式
objects = [{"name": "chair"}, {"name": "table"}, {"name": "door"}]

# 带距离时（可选）
objects = [{"name": "chair", "distance": 2.5}]
```

**修改位置：**

1. `models/model_manager.py:653-658` - VLM解析输出时
2. `agents/perception_agent.py:176-186` - 格式化输出时
3. `agents/decision_agent.py:610-611, 792-793` - 删除fallback

### 2. 删除fallback逻辑

**修改前：**
```python
objects = [o.get("object", o.get("name", str(o))) for o in objects_raw[:3]]
visible_objects = [{"name": o} if isinstance(o, str) else o for o in (objects or [])]
```

**修改后：**
```python
objects = [o.get("name", "unknown") for o in objects_raw[:3]]
visible_objects = objects  # 直接使用，无需转换
```

### 3. 改进汇总日志

**在每step结束后添加输出摘要：**

```python
# run_vln_experiment.py 在agent调用完成后
self.logger.info(f"""
=== Step {step} Summary ===
Perception: room={room_type}, objects={[o.get('name') for o in objects[:3]]}, hint={nav_hint[:40]}
Trajectory: traveled={distance_traveled:.1f}m, to_goal={distance_to_goal:.1f}m, heading={heading}
Decision: actions={len(actions)}, confidence={confidence:.2f}
=== End ===
""")
```

### 4. 错误高亮

**在agent调用失败时打印醒目标记：**

```python
# run_vln_experiment.py 在catch异常时
self.logger.error(f"!!! AGENT FAILED: {agent_name} !!!")
self.logger.error(f"!!! Error: {str(e)} !!!")
self.logger.error(f"!!! Step: {step}, Position: {position} !!!")
```

**调试时grep快速定位：**
```bash
grep "!!!" experiment.log
```

## 测试验证

### 验证方法

运行评估脚本：
```bash
python run_vln_experiment.py --use-remote-llm --llm-server http://localhost:8000 --episodes 3 2>&1 | tee test_log.txt
```

### 成功指标

1. 日志中出现统一的输出摘要格式
2. 无"object"字段的fallback逻辑残留
3. 错误时能看到醒目的"!!!"标记
4. 所有objects字段统一为"name"

### 验证命令

```bash
# 检查无fallback残留
grep "o.get.*object.*o.get.*name" agents/decision_agent.py
# 期望：无输出

# 检查汇总日志格式
grep "=== Step" test_log.txt | head -5
# 期望：看到Perception/Trajectory/Decision汇总
```

## 影响范围

- 只修改4个文件，约60行改动
- 不改变agent核心逻辑
- 不引入新依赖
- 向后兼容（只统一字段名，不删除字段）

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 遗漏某处使用"object"字段 | grep全局搜索确认 |
| 日志改动影响性能 | 汇总日志只在每step打印一次 |
| 旧数据兼容性 | 保留distance等可选字段 |

## Why

实验中数据格式不一致导致代码冗余（fallback逻辑），日志分散导致调试困难。直接统一格式和改进日志是最小改动、最大收益方案。

## How to apply

在各输出位置统一使用"name"字段，删除fallback逻辑，添加汇总日志和错误高亮。