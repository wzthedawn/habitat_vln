#!/usr/bin/env python3
"""测试不同的四元数转换方式"""

import habitat_sim
import numpy as np
import json
import os
import quaternion

# 加载 R2R 数据
with open('/data/WZ/Dataset/R2R_VLNCE_v1-3/val_seen/val_seen.json') as f:
    data = json.load(f)

ep1 = data['episodes'][0]
scene_full = ep1['scene_id']
scene_name = scene_full.split('/')[1]
start_pos = ep1['start_position']
start_rot = ep1['start_rotation']  # [x, y, z, w]
ref_path = ep1['reference_path']

# 期望的方向（从参考路径）
dx = ref_path[1][0] - ref_path[0][0]
dz = ref_path[1][2] - ref_path[0][2]
expected_forward = np.array([dx, 0, dz])
expected_forward = expected_forward / np.linalg.norm(expected_forward)
print(f"期望的朝向 (归一化): ({expected_forward[0]:.3f}, {expected_forward[1]:.3f}, {expected_forward[2]:.3f})")

# 测试不同的四元数转换
test_cases = [
    ("直接 [x,y,z,w]", start_rot),
    ("取反 [-x,-y,-z,-w]", [-start_rot[0], -start_rot[1], -start_rot[2], -start_rot[3]]),
    (" negate Y [x,-y,z,w]", [start_rot[0], -start_rot[1], start_rot[2], start_rot[3]]),
    ("绕 Y 旋转 180°", [-start_rot[0], start_rot[1], -start_rot[2], start_rot[3]]),
]

for name, rot in test_cases:
    q = np.quaternion(rot[3], rot[0], rot[1], rot[2])  # (w, x, y, z)
    forward = quaternion.rotate_vectors(q, np.array([0.0, 0.0, -1.0]))
    dot = np.dot(forward, expected_forward)
    angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    print(f"{name}: forward=({forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}), 角度误差={angle:.1f}°")

# 现在测试用 habitat-sim 直接渲染
scene_path = f"/data/WZ/Dataset/mp3d_dataset/mp3d/{scene_name}/{scene_name}.glb"

color_sensor_spec = habitat_sim.CameraSensorSpec()
color_sensor_spec.uuid = "rgb"
color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
color_sensor_spec.resolution = [480, 640]
color_sensor_spec.position = np.array([0.0, 1.5, 0.0])

agent_cfg = habitat_sim.AgentConfiguration(
    height=1.5,
    radius=0.1,
    sensor_specifications=[color_sensor_spec],
)

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.gpu_device_id = 0

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(0)

print("\n=== 渲染测试 ===")

for name, rot in test_cases:
    state = agent.get_state()
    state.position = np.array(start_pos)
    state.rotation = np.array(rot, dtype=np.float32)
    agent.set_state(state)

    obs = sim.get_sensor_observations(0)
    rgb_image = obs['rgb']

    from PIL import Image
    img = Image.fromarray(rgb_image)
    safe_name = name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '')
    img.save(f"/home/WZ/MA_VLN/habitat_vln/test_output/test_{safe_name}.png")
    print(f"保存：test_{safe_name}.png")

sim.close()
print("\n完成！")
