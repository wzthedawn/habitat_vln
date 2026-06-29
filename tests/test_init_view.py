#!/usr/bin/env python3
"""测试 R2R episode 1 初始位置图像"""

import habitat_sim
import numpy as np
import json
import os

# 加载 R2R 数据 - 使用 /data/WZ/Dataset/r2r 数据集
with open('/data/WZ/Dataset/r2r/v1/val_seen/val_seen.json') as f:
    data = json.load(f)

ep1 = data['episodes'][0]
# scene_id: mp3d/s8pcmisQ38h/s8pcmisQ38h.glb -> 提取 s8pcmisQ38h
scene_name = ep1['scene_id'].split('/')[1]
start_pos = ep1['start_position']
start_rot = ep1['start_rotation']  # [x, y, z, w]

print(f"Dataset: /data/WZ/Dataset/r2r/v1/val_seen/val_seen.json")
print(f"Scene: {scene_name}")
print(f"Start position: {start_pos}")
print(f"Start rotation [x,y,z,w]: {start_rot}")

# 场景路径
scene_path = f"/data/WZ/Dataset/mp3d_dataset/mp3d/{scene_name}/{scene_name}.glb"
print(f"Scene path: {scene_path}")

# 配置传感器
color_sensor_spec = habitat_sim.CameraSensorSpec()
color_sensor_spec.uuid = "rgb"
color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
color_sensor_spec.resolution = [480, 640]
color_sensor_spec.position = np.array([0.0, 1.5, 0.0])

depth_sensor_spec = habitat_sim.CameraSensorSpec()
depth_sensor_spec.uuid = "depth"
depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
depth_sensor_spec.resolution = [480, 640]
depth_sensor_spec.position = np.array([0.0, 1.5, 0.0])

# 配置 Agent
agent_cfg = habitat_sim.AgentConfiguration(
    height=1.5,
    radius=0.1,
    sensor_specifications=[color_sensor_spec, depth_sensor_spec],
)

# 配置 Simulator
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.gpu_device_id = 0

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

# 获取 Agent 并设置初始状态
agent = sim.initialize_agent(0)
state = agent.get_state()

# 设置位置
state.position = np.array(start_pos)

# 设置旋转 - 直接使用 R2R 的 [x,y,z,w]
state.rotation = np.array(start_rot, dtype=np.float32)

agent.set_state(state)

# 获取观测
obs = sim.get_sensor_observations(0)
rgb_image = obs['rgb']

# 保存图像
output_dir = "/home/WZ/MA_VLN/habitat_vln/test_output"
os.makedirs(output_dir, exist_ok=True)

from PIL import Image
img = Image.fromarray(rgb_image)
img.save(f"{output_dir}/episode1_init_view.png")
print(f"\n图像已保存到：{output_dir}/episode1_init_view.png")

# 打印相机朝向信息
import quaternion
q = np.quaternion(start_rot[3], start_rot[0], start_rot[1], start_rot[2])
forward_local = np.array([0.0, 0.0, -1.0])
forward_world = quaternion.rotate_vectors(q, forward_local)
print(f"\n相机朝向（世界坐标系）: ({forward_world[0]:.3f}, {forward_world[1]:.3f}, {forward_world[2]:.3f})")

# 计算参考路径方向
ref_path = ep1['reference_path']
dx = ref_path[1][0] - ref_path[0][0]
dz = ref_path[1][2] - ref_path[0][2]
print(f"参考路径方向：dx={dx:.3f}, dz={dz:.3f}")

sim.close()
