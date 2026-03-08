"""Observation processor for handling visual data."""

from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np

from core.context import VisualFeatures


class ObservationProcessor:
    """
    Observation Processor for handling visual data.

    Processes:
    - RGB images
    - Depth maps
    - Panoramic views
    - Semantic segmentation
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize observation processor.

        Args:
            config: Processor configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("ObservationProcessor")

        # Processing settings
        self.image_size = self.config.get("image_size", (224, 224))
        self.normalize_depth = self.config.get("normalize_depth", True)
        self.depth_min = self.config.get("depth_min", 0.0)
        self.depth_max = self.config.get("depth_max", 10.0)

        # Visual encoder (lazy loaded)
        self._visual_encoder = None

    def process(self, observation: Dict[str, Any]) -> VisualFeatures:
        """
        Process observation into visual features.

        Args:
            observation: Raw observation dictionary

        Returns:
            VisualFeatures object
        """
        features = VisualFeatures()

        # Process RGB
        if "rgb" in observation:
            features.rgb_embedding = self._process_rgb(observation["rgb"])

        # Process depth
        if "depth" in observation:
            features.depth_embedding = self._process_depth(observation["depth"])

        # Process panoramic view if available
        if any(k.startswith("panoramic") for k in observation):
            features.pano_features = self._process_panoramic(observation)

        return features

    def _process_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """
        Process RGB image.

        Args:
            rgb: RGB image array

        Returns:
            Processed image or embedding
        """
        # Resize if needed
        if rgb.shape[:2] != self.image_size:
            rgb = self._resize_image(rgb, self.image_size)

        # Normalize
        rgb = rgb.astype(np.float32) / 255.0

        return rgb

    def _process_depth(self, depth: np.ndarray) -> Dict[str, Any]:
        """
        Process depth map.

        Args:
            depth: Depth map array

        Returns:
            Processed depth features
        """
        if self.normalize_depth:
            depth = np.clip(depth, self.depth_min, self.depth_max)
            depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)

        features = {
            "depth_map": depth,
            "mean_depth": float(np.mean(depth)),
            "min_depth": float(np.min(depth)),
            "max_depth": float(np.max(depth)),
        }

        return features

    def _process_panoramic(
        self, observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process panoramic observations.

        Args:
            observation: Observation with panoramic data

        Returns:
            Panoramic features
        """
        pano_features = {}

        # Collect panoramic views
        views = []
        for key in observation:
            if key.startswith("panoramic_"):
                views.append(observation[key])

        if views:
            pano_features["num_views"] = len(views)
            pano_features["views"] = views

        return pano_features

    def _resize_image(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize image to target size."""
        try:
            import cv2
            return cv2.resize(image, target_size[::-1])
        except ImportError:
            # Simple resize fallback
            self.logger.warning("cv2 not available, skipping resize")
            return image

    def extract_features(
        self, observation: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract visual features using encoder.

        Args:
            observation: Observation dictionary

        Returns:
            Feature vector
        """
        # Lazy load encoder
        if self._visual_encoder is None:
            try:
                from models.visual_encoder import VisualEncoder
                self._visual_encoder = VisualEncoder(self.config.get("encoder", {}))
            except ImportError:
                self.logger.warning("VisualEncoder not available")
                return None

        if "rgb" in observation:
            return self._visual_encoder.encode_image(observation["rgb"])

        return None

    def detect_objects(
        self, observation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in observation.

        Args:
            observation: Observation dictionary

        Returns:
            List of detected objects
        """
        # Placeholder for object detection
        # In practice, integrate with object detection models
        return []

    def get_obstacle_map(
        self, depth: np.ndarray, threshold: float = 1.0
    ) -> np.ndarray:
        """
        Generate obstacle map from depth.

        Args:
            depth: Depth map
            threshold: Depth threshold for obstacles

        Returns:
            Binary obstacle map
        """
        return (depth < threshold).astype(np.uint8)

    def compute_collision(
        self, depth: np.ndarray, collision_threshold: float = 0.2
    ) -> bool:
        """
        Check for collision from depth map.

        Args:
            depth: Depth map
            collision_threshold: Threshold for collision detection

        Returns:
            True if collision detected
        """
        # Check center region of depth map
        h, w = depth.shape
        center_region = depth[h//3:2*h//3, w//3:2*w//3]

        min_depth = np.min(center_region)

        return min_depth < collision_threshold


class PanoramaBuilder:
    """
    Builder for creating panoramic observations.
    """

    def __init__(self, num_views: int = 12):
        """
        Initialize panorama builder.

        Args:
            num_views: Number of views for panorama
        """
        self.num_views = num_views
        self.angle_per_view = 360.0 / num_views

    def build_from_env(
        self, env: Any, agent_id: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Build panorama from environment.

        Args:
            env: Habitat environment
            agent_id: Agent ID

        Returns:
            Dictionary of panoramic views
        """
        views = {}

        if hasattr(env, "_sim") and env._sim is not None:
            agent = env._sim.get_agent(agent_id)
            original_rotation = agent.get_state().rotation

            for i in range(self.num_views):
                # Rotate agent
                angle = i * self.angle_per_view
                self._set_agent_rotation(agent, angle)

                # Get observation
                obs = env._sim.get_sensor_observations()
                views[f"panoramic_{i}"] = obs.get("rgb", np.zeros((224, 224, 3)))

            # Restore rotation
            agent.set_state(original_rotation)

        return views

    def _set_agent_rotation(self, agent: Any, angle_deg: float) -> None:
        """Set agent rotation."""
        try:
            import habitat_sim
            import numpy as np

            angle_rad = np.deg2rad(angle_deg)
            rotation = habitat_sim.utils.quat_from_angle_axis(
                angle_rad, np.array([0, 1, 0])
            )
            state = agent.get_state()
            state.rotation = rotation
            agent.set_state(state)

        except ImportError:
            pass


class DepthProcessor:
    """Processor specifically for depth maps."""

    @staticmethod
    def normalize(
        depth: np.ndarray,
        min_val: float = 0.0,
        max_val: float = 10.0,
    ) -> np.ndarray:
        """Normalize depth values."""
        depth = np.clip(depth, min_val, max_val)
        return (depth - min_val) / (max_val - min_val)

    @staticmethod
    def compute_point_cloud(
        depth: np.ndarray,
        fov: float = 90.0,
    ) -> np.ndarray:
        """
        Compute point cloud from depth map.

        Args:
            depth: Depth map (H, W)
            fov: Field of view in degrees

        Returns:
            Point cloud (N, 3)
        """
        h, w = depth.shape
        fov_rad = np.deg2rad(fov)

        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w]
        y = (y - h / 2) / h
        x = (x - w / 2) / w

        # Compute 3D coordinates
        z = depth
        x_3d = x * z * np.tan(fov_rad / 2)
        y_3d = y * z * np.tan(fov_rad / 2)

        # Stack into point cloud
        points = np.stack([x_3d, y_3d, z], axis=-1)

        return points.reshape(-1, 3)

    @staticmethod
    def find_floor_plane(
        depth: np.ndarray, threshold: float = 0.5
    ) -> Optional[float]:
        """
        Find floor plane distance from depth.

        Args:
            depth: Depth map
            threshold: Variance threshold for floor detection

        Returns:
            Floor distance or None
        """
        # Check bottom portion of depth map
        h = depth.shape[0]
        bottom_depth = depth[-h//4:, :]

        if np.var(bottom_depth) < threshold:
            return float(np.mean(bottom_depth))

        return None