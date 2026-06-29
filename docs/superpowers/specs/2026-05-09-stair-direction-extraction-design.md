---
name: stair-direction-extraction-fix
description: Fix VLM not outputting stair_direction JSON field in perception_agent.py
type: project
---

# Stair Direction Extraction Fix

## Problem

VLM (Qwen3.6-35B-A3B) detects `room_type: "stairs"` but does not output the required `stair_direction` JSON field. Instead, it describes stair direction in `scene_description` text (e.g., "The stairs curve slightly to the left as they ascend").

**Evidence from experiment:**
- Prompt requires `"stair_direction":"up/down/none"` in JSON output
- VLM outputs `room_type: "stairs"` correctly
- VLM describes "as they ascend" in scene_description
- No `stair_direction` field in JSON response

**Root cause:**
Prompt Guidelines uses conditional phrasing "ONLY when stairs are visible", which VLM may interpret as optional field rather than mandatory.

## Solution

Modify RGB+Depth prompt in `perception_agent.py` to emphasize `stair_direction` is a MANDATORY field.

### Changes

**File:** `agents/perception_agent.py`
**Section:** `_analyze_with_vlm()` method, RGB+Depth prompt (lines ~285-302)

**Before:**
```
## Required Output (JSON only)
{"room_type":"...", "scene_description":"...", "objects":[...], "stair_direction":"up/down/none"}

## Guidelines
4. stair_direction: ONLY when stairs are visible:
   - "up" if stairs ascend away from you
   - "down" if stairs descend away from you
   - "none" if no stairs visible
```

**After:**
```
## Required Output (ALL fields mandatory, no fields can be omitted)
{"room_type":"...", "scene_description":"...", "objects":[...], "stair_direction":"up/down/none"}

## Guidelines
4. stair_direction: MANDATORY field - always include in JSON:
   - "up" if stairs ascend away from viewer
   - "down" if stairs descend away from viewer
   - "none" if no stairs in scene (output this value, do not omit the field)
```

### Why this works

- Removes conditional "ONLY when" phrasing that suggests optional
- Explicitly states "ALL fields mandatory"
- Clarifies "none" must be output even when no stairs visible
- Uses stronger language "MANDATORY" and "always include"

## Implementation Plan

1. Edit `agents/perception_agent.py` line ~285 and ~299-302
2. Test with single step experiment on stairs scene
3. Verify VLM output contains `stair_direction` field

## Testing

Run episode 1 (stairs task) for 5-10 steps and check:
- `perception_output` JSON contains `stair_direction` field
- Value is "up" or "down" (not "none") when room_type is "stairs"

## Impact

- Single file change: `agents/perception_agent.py`
- ~5 lines modified in prompt text
- No code logic changes
- Backward compatible with existing regex fallback