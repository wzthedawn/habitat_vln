#!/usr/bin/env python3
"""Test YOLO detection fix."""
import sys
sys.path.insert(0, '/root/habitat_vln')

from models.model_manager import ModelManager

# Create a new instance (bypass singleton)
config = {"use_remote": False, "device": "cuda"}
mm = object.__new__(ModelManager)
mm.config = config
import logging
mm.logger = logging.getLogger("ModelManager")
mm.device = "cuda"
mm.use_int8 = True
mm.use_remote = False
mm.remote_server_url = "http://localhost:8000"
mm._remote_client = None
mm._remote_healthy = False
mm._models = {}
mm._tokenizers = {}
mm._initialized = True
mm._models_loaded = False
mm._llm_loaded = False

# Load YOLO
mm._load_yolo()
yolo_model = mm.get_model("yolov5s")
print(f"YOLO model loaded: {yolo_model is not None}")

# Test detection
from PIL import Image
import numpy as np

img = Image.open("/tmp/yolo_debug.jpg")
rgb_image = np.array(img)

print("Testing detect_objects with conf=0.05...")
detections = mm.detect_objects(rgb_image, confidence_threshold=0.05)
print(f"Detections: {len(detections)}")
for det in detections[:5]:
    print(f"  {det['name']}: {det['confidence']:.3f}")