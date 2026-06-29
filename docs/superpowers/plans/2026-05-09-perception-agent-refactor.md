# PerceptionAgent 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构perception_agent.py - 按职责分组重组、精简代码、中文注释

**Architecture:** 单文件重构，5个职责分组，不改变任何逻辑

**Tech Stack:** Python, 代码重组

---

## File Structure

| File | Responsibility | Change Type |
|------|---------------|-------------|
| `agents/perception_agent.py` | VLM感知智能体 | 重构重组 |

---

### Task 1: 重写文件头部和类常量部分

**Files:**
- Modify: `agents/perception_agent.py:1-60`

将文件头部docstring、类定义、常量部分改为中文并精简。

- [ ] **Step 1: 重写文件头部docstring和类定义**

替换当前第1-60行，使用中文注释和精简的类常量定义：

```python
"""感知智能体 - 使用VLM分析视觉观测

核心功能：
- RGB+Depth图像融合分析
- 物体检测与场景描述
- 楼梯/障碍物检测
- 房间分类

纯VLM方案，无YOLO、无规则检测。
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext


# ============================================================
# 第一部分：配置与初始化
# ============================================================


class PerceptionAgent(BaseAgent):
    """感知智能体 - 负责视觉感知分析

    使用VLM（Qwen3.6）完成所有感知任务：
    - RGB+Depth图像融合
    - 物体检测与场景描述
    - 楼梯方向识别
    - 房间分类
    """

    # 房间类型关键词映射
    ROOM_KEYWORDS = {
        "bedroom": ["bed", "nightstand", "dresser", "wardrobe", "pillow", "blanket"],
        "bathroom": ["toilet", "sink", "shower", "bathtub", "towel", "mirror"],
        "kitchen": ["refrigerator", "stove", "oven", "sink", "counter", "cabinet", "microwave"],
        "living_room": ["sofa", "couch", "television", "tv", "coffee table", "fireplace", "armchair"],
        "dining_room": ["dining table", "chair", "sideboard"],
        "hallway": ["door", "corridor", "passage", "wall"],
        "stairs": ["stairs", "staircase", "steps", "railing", "banister", "handrail"],
        "office": ["desk", "computer", "chair", "bookshelf", "monitor"],
        "garage": ["car", "tool", "workbench"],
    }

    # 导航相关物体（用于路径规划）
    NAVIGATION_OBJECTS = {
        "door", "stairs", "staircase", "steps", "corridor", "hallway",
        "entrance", "exit", "wall", "floor", "ceiling",
    }

    # 地标物体（用于定位参照）
    LANDMARK_OBJECTS = {
        "chair", "table", "desk", "bed", "sofa", "couch",
        "cabinet", "shelf", "bookshelf", "refrigerator", "tv",
        "piano", "bench", "rug", "carpet",
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("PerceptionAgent")
        self.max_objects = self.config.get("max_objects", 10)
        self._model_manager = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "perception_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.PERCEPTION

    def get_required_inputs(self) -> List[str]:
        return ["rgb_image", "depth_image"]

    def get_output_keys(self) -> List[str]:
        return ["room_type", "objects", "landmarks", "scene_description"]

    def initialize(self) -> None:
        """初始化模型管理器"""
        if self._initialized:
            return
        try:
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)
            self._initialized = True
            self.logger.info("感知智能体初始化完成（RGB+Depth模式）")
        except Exception as e:
            self.logger.warning(f"模型管理器初始化失败: {e}")
            self._initialized = True

    # 默认输出值（当VLM无响应时使用）
    DEFAULT_WALKABLE = {"left": {"clear": True}, "center": {"clear": True}, "right": {"clear": True}, "recommended": "center"}
    DEFAULT_OBSTACLE = {"blocked": False, "min_distance": 5.0}
```

- [ ] **Step 2: Commit**

```bash
git add agents/perception_agent.py
git commit -m "refactor: 重构perception_agent头部和初始化部分
- 中文注释
- 精简DEFAULT常量
- 清晰的职责分组注释"
```

---

### Task 2: 重构核心处理流程部分

**Files:**
- Modify: `agents/perception_agent.py:107-370` (process, _get_rgb/depth, _analyze_with_vlm)

- [ ] **Step 1: 在initialize方法后添加分组注释，重写process方法开头**

在initialize方法后添加核心处理流程分组注释，并精简process方法中的注释。

添加分组注释：
```python

    # ============================================================
    # 第二部分：核心处理流程
    # ============================================================
```

将process方法中的英文注释改为中文，精简冗余日志。

- [ ] **Step 2: 重写_analyze_with_vlm方法的prompt注释**

将VLM prompt中的英文注释改为中文，保持prompt内容不变。

- [ ] **Step 3: Commit**

```bash
git add agents/perception_agent.py
git commit -m "refactor: 重构核心处理流程部分
- 中文注释
- 精简冗余日志"
```

---

### Task 3: 重构VLM响应解析部分

**Files:**
- Modify: `agents/perception_agent.py:380-700` (_parse_vlm_response, _extract_from_json, etc.)

- [ ] **Step 1: 在_analyze_with_vlm后添加解析分组注释**

```python

    # ============================================================
    # 第三部分：VLM响应解析
    # ============================================================
```

- [ ] **Step 2: 重写解析方法的docstring为中文**

将以下方法的docstring改为中文：
- `_parse_vlm_response` - 解析VLM响应
- `_validate_perception_output` - 验证输出完整性
- `_extract_from_json` - 从JSON提取字段
- `_try_partial_json_extraction` - 部分JSON提取
- `_regex_fallback_extract` - 正则fallback提取

- [ ] **Step 3: 精简解析方法中的冗余代码**

合并相似的日志输出，删除重复的warning。

- [ ] **Step 4: Commit**

```bash
git add agents/perception_agent.py
git commit -m "refactor: 重构VLM响应解析部分
- 中文docstring
- 精简冗余日志"
```

---

### Task 4: 重构辅助方法和Debate接口部分

**Files:**
- Modify: `agents/perception_agent.py:700-1004`

- [ ] **Step 1: 在解析部分后添加辅助方法分组注释**

```python

    # ============================================================
    # 第四部分：辅助方法
    # ============================================================
```

将辅助方法docstring改为中文：
- `_normalize_room_type`
- `_infer_room_from_objects`
- `_match_landmarks`

- [ ] **Step 2: 在辅助方法后添加Debate接口分组注释**

```python

    # ============================================================
    # 第五部分：Debate接口（多智能体协商）
    # ============================================================
```

将Debate方法docstring改为中文：
- `build_debate_opinion`
- `_build_perception_opinion_with_llm`
- `_parse_perception_opinion_response`
- `_build_perception_opinion_fallback`

- [ ] **Step 3: Commit**

```bash
git add agents/perception_agent.py
git commit -m "refactor: 重构辅助方法和Debate接口
- 中文注释
- 清晰的分组结构"
```

---

### Task 5: 最终验证

- [ ] **Step 1: 运行语法检查**

```bash
python -m py_compile agents/perception_agent.py
```

Expected: 无语法错误

- [ ] **Step 2: 检查文件行数**

```bash
wc -l agents/perception_agent.py
```

Expected: 约900行（比原来少约100行）

---

## Self-Review

**1. Spec coverage:** ✅
- 职责分组重组：Task 1-4
- 中文注释：所有Task
- 精简代码：Task 2-3

**2. Placeholder scan:** ✅
- 无TBD/TODO
- 所有代码片段完整

**3. Type consistency:** ✅
- 单文件重构，无跨文件类型依赖