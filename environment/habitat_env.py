"""Habitat environment wrapper for VLN navigation."""

from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np

from core.context import NavContext, NavContextBuilder, VisualFeatures
from core.action import Action, ActionType


class HabitatEnv:
    """
    Habitat Environment wrapper for VLN navigation.

    Provides a unified interface for interacting with Habitat-sim
    for vision-language navigation tasks.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Habitat environment.

        Args:
            config: Environment configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("HabitatEnv")

        # Environment settings
        self.scene_id = self.config.get("scene_id", "")
        self.max_episode_steps = self.config.get("max_episode_steps", 500)
        self.sensor_height = self.config.get("sensor_height", 1.5)
        self.turn_angle = self.config.get("turn_angle", 15.0)

        # Habitat components (lazy loaded)
        self._env = None
        self._sim = None
        self._initialized = False

        # Episode state
        self._episode_started = False
        self._current_step = 0

    def initialize(self) -> None:
        """Initialize the Habitat environment."""
        if self._initialized:
            return

        try:
            self._initialize_habitat()
            self._initialized = True
            self.logger.info("Habitat environment initialized")

        except ImportError as e:
            self.logger.warning(f"Habitat not available: {e}, using mock")
            self._initialized = True

        except Exception as e:
            self.logger.error(f"Failed to initialize Habitat: {e}")
            self._initialized = True  # Use mock mode

    def _initialize_habitat(self) -> None:
        """Initialize actual Habitat environment using habitat_sim directly."""
        import habitat_sim

        # Create RGB sensor specification
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [self.config.get("image_height", 256),
                                   self.config.get("image_width", 256)]
        rgb_sensor.position = [0.0, self.sensor_height, 0.0]

        # Create Depth sensor specification (optional)
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [self.config.get("image_height", 256),
                                    self.config.get("image_width", 256)]
        depth_sensor.position = [0.0, self.sensor_height, 0.0]

        # Define action space
        forward_amount = self.config.get("forward_amount", 0.25)

        # Agent configuration
        agent_cfg = habitat_sim.AgentConfiguration(
            height=self.sensor_height,
            radius=0.1,
            sensor_specifications=[rgb_sensor, depth_sensor],
            action_space={
                "move_forward": habitat_sim.ActionSpec(
                    "move_forward",
                    habitat_sim.ActuationSpec(amount=forward_amount)
                ),
                "turn_left": habitat_sim.ActionSpec(
                    "turn_left",
                    habitat_sim.ActuationSpec(amount=self.turn_angle)
                ),
                "turn_right": habitat_sim.ActionSpec(
                    "turn_right",
                    habitat_sim.ActuationSpec(amount=self.turn_angle)
                ),
                "look_up": habitat_sim.ActionSpec(
                    "look_up",
                    habitat_sim.ActuationSpec(amount=15.0)
                ),
                "look_down": habitat_sim.ActionSpec(
                    "look_down",
                    habitat_sim.ActuationSpec(amount=15.0)
                ),
                "stop": habitat_sim.ActionSpec(
                    "stop",
                    habitat_sim.ActuationSpec(amount=0.0)
                ),
            }
        )

        # Simulator configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_id
        sim_cfg.enable_physics = False
        sim_cfg.allow_sliding = True

        # Create configuration and simulator
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)

        # Recompute navmesh if needed
        if not self._sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = self.sensor_height
            navmesh_settings.agent_radius = 0.1
            self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings)

        self.logger.info(f"Habitat simulator initialized with scene: {self.scene_id}")

    def reset(self) -> NavContext:
        """
        Reset environment and return initial observation.

        Returns:
            Initial navigation context
        """
        self.initialize()

        self._current_step = 0
        self._episode_started = True

        if self._sim is not None:
            # Get initial observation from simulator
            obs = self._get_observation()
            return self._observation_to_context(obs)

        # Mock mode
        return self._create_mock_context()

    def step(self, action: Action) -> Tuple[NavContext, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (context, reward, done, info)
        """
        if not self._episode_started:
            self.logger.warning("Episode not started, call reset() first")
            return self._create_mock_context(), 0.0, True, {}

        self._current_step += 1

        if self._sim is not None:
            return self._habitat_sim_step(action)

        # Mock mode
        return self._mock_step(action)

    def _get_observation(self) -> Dict:
        """Get current observation from simulator."""
        if self._sim is None:
            return {}

        obs = {}
        try:
            # Get RGB and depth observations
            obs["rgb"] = self._sim.get_sensor_observations()["rgb"]
            obs["depth"] = self._sim.get_sensor_observations()["depth"]
        except Exception as e:
            self.logger.warning(f"Failed to get observation: {e}")

        return obs

    def _habitat_sim_step(self, action: Action) -> Tuple[NavContext, float, bool, Dict]:
        """Execute step using habitat_sim directly."""
        # Convert action to habitat_sim action
        action_map = {
            ActionType.MOVE_FORWARD: "move_forward",
            ActionType.TURN_LEFT: "turn_left",
            ActionType.TURN_RIGHT: "turn_right",
            ActionType.LOOK_UP: "look_up",
            ActionType.LOOK_DOWN: "look_down",
            ActionType.STOP: "stop",
        }

        sim_action = action_map.get(action.action_type, "move_forward")

        # Execute action in simulator
        obs = self._sim.step(sim_action)

        # Convert observation to context
        context = self._observation_to_context(obs)

        # Check if done
        done = self._current_step >= self.max_episode_steps

        info = {
            "step": self._current_step,
            "success": False,  # Will be computed externally
        }

        return context, 0.0, done, info

    def _mock_step(self, action: Action) -> Tuple[NavContext, float, bool, Dict]:
        """Execute mock step for testing."""
        # Create updated context
        context = self._create_mock_context()
        context.step_count = self._current_step
        context.add_action(action)

        done = self._current_step >= self.max_episode_steps

        return context, 0.0, done, {"step": self._current_step}

    def _observation_to_context(self, obs: Dict) -> NavContext:
        """Convert Habitat observation to NavContext."""
        builder = NavContextBuilder()

        # Extract visual features
        visual_features = VisualFeatures()

        if "rgb" in obs:
            visual_features.rgb_embedding = obs["rgb"]

        if "depth" in obs:
            visual_features.depth_embedding = obs["depth"]

        # Get agent state
        if self._sim is not None:
            agent = self._sim.get_agent(0)
            state = agent.get_state()

            builder.with_position(tuple(state.position))

            # Calculate yaw from quaternion
            # rotation is a quaternion [x, y, z, w]
            import math
            rotation = state.rotation
            if hasattr(rotation, '__len__') and len(rotation) >= 4:
                # Quaternion to euler yaw
                q = rotation
                siny_cosp = 2 * (q[0] * q[1] + q[3] * q[2])
                cosy_cosp = q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2]
                yaw = math.atan2(siny_cosp, cosy_cosp)
            else:
                yaw = 0.0
            builder.with_rotation(yaw)

        context = builder.build()
        context.visual_features = visual_features
        context.step_count = self._current_step

        return context

    def _create_mock_context(self) -> NavContext:
        """Create mock context for testing."""
        builder = NavContextBuilder()
        builder.with_instruction("Mock navigation instruction")
        builder.with_position((0.0, 0.0, 0.0))

        context = builder.build()
        context.visual_features = VisualFeatures()

        return context

    def _calculate_reward(self, obs: Dict) -> float:
        """Calculate reward from observation."""
        # Default reward calculation
        if obs.get("success", False):
            return 1.0

        return -0.01  # Small step penalty

    def render(self, mode: str = "rgb") -> Optional[np.ndarray]:
        """
        Render environment.

        Args:
            mode: Render mode (rgb, depth, etc.)

        Returns:
            Rendered image or None
        """
        if self._env is not None:
            obs = self._env.sim.get_sensor_observations()
            return obs.get(mode)

        return None

    def close(self) -> None:
        """Close environment."""
        if self._env is not None:
            self._env.close()

        self._env = None
        self._sim = None
        self._initialized = False

    def get_agent_position(self) -> Tuple[float, float, float]:
        """Get current agent position."""
        if self._sim is not None:
            agent = self._sim.get_agent(0)
            state = agent.get_state()
            return tuple(state.position.tolist())

        return (0.0, 0.0, 0.0)

    def get_agent_rotation(self) -> float:
        """Get current agent rotation (yaw)."""
        if self._sim is not None:
            agent = self._sim.get_agent(0)
            state = agent.get_state()
            return state.rotation.tolist()[1]

        return 0.0

    def set_agent_state(
        self,
        position: Tuple[float, float, float],
        rotation: float,
    ) -> None:
        """Set agent state."""
        if self._sim is not None:
            import habitat_sim
            agent = self._sim.get_agent(0)
            state = habitat_sim.AgentState()
            state.position = position
            state.rotation = habitat_sim.utils.quat_from_angle_axis(
                rotation, np.array([0, 1, 0])
            )
            agent.set_state(state)

    def get_navigable_points(self) -> List[Tuple[float, float, float]]:
        """Get list of navigable points in environment."""
        if self._sim is not None:
            # Sample random navigable points
            points = []
            for _ in range(100):  # Sample 100 points
                point = self._sim.pathfinder.get_random_navigable_point()
                points.append(tuple(point.tolist()))
            return points

        return []

    def is_navigable(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is navigable."""
        if self._sim is not None:
            return self._sim.pathfinder.is_navigable(position)

        return True

    def get_geodesic_distance(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """Calculate geodesic distance between points."""
        if self._sim is not None:
            return self._sim.geodesic_distance(start, end)

        # Euclidean distance as fallback
        return np.sqrt(sum((s - e) ** 2 for s, e in zip(start, end)))

    @property
    def episode_over(self) -> bool:
        """Check if episode is over."""
        if self._env is not None:
            return self._env.episode_over

        return self._current_step >= self.max_episode_steps

    @property
    def current_step(self) -> int:
        """Get current step count."""
        return self._current_step


class HabitatEnvBuilder:
    """Builder for HabitatEnv instances."""

    def __init__(self):
        self._config: Dict[str, Any] = {}

    def with_scene(self, scene_id: str) -> "HabitatEnvBuilder":
        """Set scene ID."""
        self._config["scene_id"] = scene_id
        return self

    def with_max_steps(self, max_steps: int) -> "HabitatEnvBuilder":
        """Set maximum episode steps."""
        self._config["max_episode_steps"] = max_steps
        return self

    def with_sensor_height(self, height: float) -> "HabitatEnvBuilder":
        """Set sensor height."""
        self._config["sensor_height"] = height
        return self

    def with_turn_angle(self, angle: float) -> "HabitatEnvBuilder":
        """Set turn angle."""
        self._config["turn_angle"] = angle
        return self

    def build(self) -> HabitatEnv:
        """Build environment instance."""
        return HabitatEnv(self._config)