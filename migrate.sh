#!/bin/bash
# VLN项目迁移脚本
# 总大小: 代码 646MB + 模型 31GB + 数据集 25GB = ~57GB
#
# 注意: 请根据实际情况修改以下变量:
#   - TARGET: 目标服务器地址
#   - 模型路径: /habitat-t1/Habitat-test1/Model
#   - 数据集路径: /habitat-t1/Habitat-test1
#
TARGET="WZ@10.95.66.199"  # 修改为目标服务器地址

echo "=========================================="
echo "VLN项目迁移脚本"
echo "目标服务器: $TARGET"
echo "=========================================="

echo ""
echo "=== Step 1: 打包代码 (646MB) ==="
cd /root/habitat_vln
tar -czvf /tmp/habitat_vln_full.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    .

echo ""
echo "=== Step 2: 打包模型 (31GB) ==="
cd /habitat-t1/Habitat-test1/Model
tar -czvf /tmp/qwen_models.tar.gz Qwen/

echo ""
echo "=== Step 3: 打包数据集 (25GB) ==="
cd /habitat-t1/Habitat-test1
tar -czvf /tmp/habitat_data.tar.gz Dataset/ Habitat/

echo ""
echo "=== Step 4: 创建目标目录 ==="
ssh $TARGET "mkdir -p /home/MA_VLN /data/WZ/models /data/WZ/habitat_data"

echo ""
echo "=== Step 5: 传输代码 ==="
scp /tmp/habitat_vln_full.tar.gz $TARGET:/home/MA_VLN/

echo ""
echo "=== Step 6: 传输模型 ==="
scp /tmp/qwen_models.tar.gz $TARGET:/data/WZ/models/

echo ""
echo "=== Step 7: 传输数据集 ==="
scp /tmp/habitat_data.tar.gz $TARGET:/data/WZ/habitat_data/

echo ""
echo "=== Step 8: 解压文件 ==="
ssh $TARGET "cd /home/MA_VLN && tar -xzf habitat_vln_full.tar.gz"
ssh $TARGET "cd /data/WZ/models && tar -xzf qwen_models.tar.gz"
ssh $TARGET "cd /data/WZ/habitat_data && tar -xzf habitat_data.tar.gz"

echo ""
echo "=========================================="
echo "=== 迁移完成 ==="
echo "=========================================="
echo ""
echo "请在目标服务器上:"
echo "1. 创建conda环境 (Habitat, habitat_py310)"
echo "2. 修改 configs/model_config.yaml 中的模型路径为:"
echo "   /data/WZ/models/Qwen/Qwen3___5-4B"
echo "3. 配置API密钥环境变量"
echo ""