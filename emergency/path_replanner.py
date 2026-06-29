"""Path Replanner for emergency navigation scenarios.

This module provides fast path replanning using A* algorithm
with dynamic cost adjustment for blocked areas.

Key features:
1. A* pathfinding with configurable heuristics
2. Dynamic cost adjustment for obstacles
3. Multiple alternative path generation
4. Path scoring and risk assessment

Performance target: <1 second response time
Note: Pure algorithm, no LLM calls.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import time
import math
import heapq
from collections import defaultdict


@dataclass
class PathResult:
    """Result of path replanning."""
    primary_path: List[Tuple[float, float, float]]  # Main recommended path
    alternative_paths: List[List[Tuple[float, float, float]]]  # Backup paths
    path_scores: List[float]  # Risk scores for each path
    total_cost: float  # Total path cost
    computation_time_ms: float  # Time taken to compute
    success: bool  # Whether replanning succeeded
    message: str = ""  # Status message


@dataclass
class PathNode:
    """Node in A* search."""
    position: Tuple[float, float, float]
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic to goal
    f_cost: float  # Total cost
    parent: Optional['PathNode'] = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost


class PathReplanner:
    """Fast path replanning using A* algorithm.

    This class provides:
    - A* pathfinding with dynamic obstacle costs
    - Multiple alternative path generation
    - Path scoring based on safety and efficiency
    - Sub-second response times

    Usage:
        replanner = PathReplanner(navmesh)
        result = replanner.replan(
            current_pos=(5.0, 0.0, 3.0),
            goal_pos=(10.0, 0.0, 15.0),
            blocked_positions=[(7.0, 0.0, 8.0)]
        )
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the path replanner.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("PathReplanner")

        # Configuration
        self._grid_resolution = self.config.get("grid_resolution", 0.5)  # meters per cell
        self._blocked_cost_multiplier = self.config.get("blocked_cost_multiplier", 10.0)
        self._diagonal_movement = self.config.get("diagonal_movement", True)
        self._max_alternatives = self.config.get("max_alternatives", 3)
        self._heuristic_weight = self.config.get("heuristic_weight", 1.0)

        # Pathfinder reference (set externally)
        self._pathfinder = None

        # Cache
        self._grid_cache = {}
        self._last_result = None

    def set_pathfinder(self, pathfinder) -> None:
        """Set the Habitat pathfinder reference.

        Args:
            pathfinder: Habitat pathfinder instance
        """
        self._pathfinder = pathfinder
        self.logger.info("Pathfinder reference set")

    def replan(
        self,
        current_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        blocked_positions: List[Tuple[float, float, float]] = None,
        blocked_areas: List[Dict] = None,
        navmesh_bounds: Tuple = None
    ) -> PathResult:
        """Replan path avoiding blocked positions.

        Args:
            current_pos: Current position (x, y, z)
            goal_pos: Goal position (x, y, z)
            blocked_positions: List of blocked world positions
            blocked_areas: List of blocked areas with center and radius
            navmesh_bounds: Optional navigation mesh bounds

        Returns:
            PathResult with primary and alternative paths
        """
        start_time = time.time()

        blocked_positions = blocked_positions or []
        blocked_areas = blocked_areas or []

        # Try using Habitat pathfinder if available
        if self._pathfinder is not None:
            result = self._replan_with_pathfinder(
                current_pos, goal_pos, blocked_positions, blocked_areas
            )
            if result.success:
                result.computation_time_ms = (time.time() - start_time) * 1000
                self._last_result = result
                return result

        # Fallback to custom A* implementation
        result = self._replan_with_astar(
            current_pos, goal_pos, blocked_positions, blocked_areas, navmesh_bounds
        )

        result.computation_time_ms = (time.time() - start_time) * 1000
        self._last_result = result

        self.logger.info(f"Replan completed in {result.computation_time_ms:.1f}ms, success={result.success}")

        return result

    def _replan_with_pathfinder(
        self,
        current_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        blocked_positions: List[Tuple[float, float, float]],
        blocked_areas: List[Dict]
    ) -> PathResult:
        """Use Habitat pathfinder for replanning.

        Args:
            current_pos: Current position
            goal_pos: Goal position
            blocked_positions: Blocked positions
            blocked_areas: Blocked areas

        Returns:
            PathResult
        """
        try:
            # Convert to Habitat format
            start = list(current_pos)
            goal = list(goal_pos)

            # Get navigable points near start and goal
            if not self._pathfinder.is_navigable(start):
                # Find nearest navigable point
                start = self._pathfinder.snap_point(start)

            if not self._pathfinder.is_navigable(goal):
                goal = self._pathfinder.snap_point(goal)

            # Find path
            path = self._pathfinder.find_path(start, goal)

            if path and len(path) > 0:
                # Check if path goes through blocked areas
                safe_path = self._check_path_safety(path, blocked_positions, blocked_areas)

                if safe_path:
                    return PathResult(
                        primary_path=[tuple(p) for p in path],
                        alternative_paths=[],
                        path_scores=[1.0],
                        total_cost=self._calculate_path_length(path),
                        success=True,
                        message="Direct path found using pathfinder"
                    )
                else:
                    # Path goes through blocked area, need custom planning
                    self.logger.info("Path goes through blocked area, using A* fallback")

        except Exception as e:
            self.logger.warning(f"Pathfinder replan failed: {e}")

        return PathResult(
            primary_path=[],
            alternative_paths=[],
            path_scores=[],
            total_cost=0,
            success=False,
            message="Pathfinder replan failed"
        )

    def _replan_with_astar(
        self,
        current_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        blocked_positions: List[Tuple[float, float, float]],
        blocked_areas: List[Dict],
        navmesh_bounds: Tuple = None
    ) -> PathResult:
        """Custom A* implementation for path replanning.

        Args:
            current_pos: Current position
            goal_pos: Goal position
            blocked_positions: Blocked positions
            blocked_areas: Blocked areas
            navmesh_bounds: Navigation bounds

        Returns:
            PathResult
        """
        # Create grid representation
        grid, grid_info = self._create_cost_grid(
            current_pos, goal_pos, blocked_positions, blocked_areas, navmesh_bounds
        )

        # Run A*
        path = self._astar_search(grid, grid_info, current_pos, goal_pos)

        if path:
            # Generate alternative paths
            alternatives = self._generate_alternatives(
                grid, grid_info, current_pos, goal_pos, primary_path=path
            )

            # Score paths
            scores = self._score_paths([path] + alternatives, blocked_positions, blocked_areas)

            return PathResult(
                primary_path=path,
                alternative_paths=alternatives,
                path_scores=scores,
                total_cost=self._calculate_path_length(path),
                computation_time_ms=0.0,  # Will be set by caller
                success=True,
                message=f"Found path with {len(path)} waypoints"
            )

        return PathResult(
            primary_path=[],
            alternative_paths=[],
            path_scores=[],
            total_cost=0,
            computation_time_ms=0.0,  # Will be set by caller
            success=False,
            message="No path found"
        )

    def _create_cost_grid(
        self,
        current_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        blocked_positions: List[Tuple[float, float, float]],
        blocked_areas: List[Dict],
        navmesh_bounds: Tuple = None
    ) -> Tuple[Dict, Dict]:
        """Create cost grid for A* search.

        Returns:
            (grid_dict, grid_info)
        """
        resolution = self._grid_resolution

        # Determine grid bounds
        if navmesh_bounds:
            min_x, min_z = navmesh_bounds[0], navmesh_bounds[1]
            max_x, max_z = navmesh_bounds[2], navmesh_bounds[3]
        else:
            # Use current and goal positions with margin
            margin = 10.0  # meters
            min_x = min(current_pos[0], goal_pos[0]) - margin
            max_x = max(current_pos[0], goal_pos[0]) + margin
            min_z = min(current_pos[2], goal_pos[2]) - margin
            max_z = max(current_pos[2], goal_pos[2]) + margin

        # Create grid info
        grid_info = {
            "resolution": resolution,
            "min_x": min_x,
            "min_z": min_z,
            "max_x": max_x,
            "max_z": max_z,
            "rows": int((max_z - min_z) / resolution) + 1,
            "cols": int((max_x - min_x) / resolution) + 1,
        }

        # Create cost grid
        grid = defaultdict(lambda: {"cost": 1.0, "blocked": False})

        # Mark blocked cells
        for bp in blocked_positions:
            cell = self._world_to_cell(bp, grid_info)
            grid[cell]["blocked"] = True
            grid[cell]["cost"] = self._blocked_cost_multiplier

            # Also mark neighbors
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    neighbor = (cell[0] + dr, cell[1] + dc)
                    if not grid[neighbor]["blocked"]:
                        grid[neighbor]["cost"] += self._blocked_cost_multiplier * 0.3

        # Mark blocked areas
        for area in blocked_areas:
            center = area.get("center", (0, 0, 0))
            radius = area.get("radius", 1.0)

            center_cell = self._world_to_cell(center, grid_info)
            radius_cells = int(radius / resolution) + 2

            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    cell = (center_cell[0] + dr, center_cell[1] + dc)
                    world_pos = self._cell_to_world(cell, grid_info)

                    # Check if within radius
                    dist = math.sqrt(
                        (world_pos[0] - center[0]) ** 2 +
                        (world_pos[2] - center[2]) ** 2
                    )

                    if dist < radius:
                        grid[cell]["blocked"] = True
                        grid[cell]["cost"] = self._blocked_cost_multiplier
                    elif dist < radius * 1.5:
                        grid[cell]["cost"] = max(grid[cell]["cost"], self._blocked_cost_multiplier * 0.5)

        return grid, grid_info

    def _world_to_cell(self, pos: Tuple[float, float, float], grid_info: Dict) -> Tuple[int, int]:
        """Convert world position to grid cell."""
        col = int((pos[0] - grid_info["min_x"]) / grid_info["resolution"])
        row = int((pos[2] - grid_info["min_z"]) / grid_info["resolution"])
        return (row, col)

    def _cell_to_world(self, cell: Tuple[int, int], grid_info: Dict) -> Tuple[float, float, float]:
        """Convert grid cell to world position."""
        x = grid_info["min_x"] + cell[1] * grid_info["resolution"]
        z = grid_info["min_z"] + cell[0] * grid_info["resolution"]
        return (x, 0.0, z)  # Y=0 as placeholder

    def _astar_search(
        self,
        grid: Dict,
        grid_info: Dict,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """A* search algorithm.

        Args:
            grid: Cost grid
            grid_info: Grid information
            start: Start position
            goal: Goal position

        Returns:
            Path as list of world positions, or empty list if not found
        """
        start_cell = self._world_to_cell(start, grid_info)
        goal_cell = self._world_to_cell(goal, grid_info)

        # Check if start or goal is blocked
        if grid[start_cell]["blocked"] or grid[goal_cell]["blocked"]:
            # Try to find nearby unblocked cells
            if grid[start_cell]["blocked"]:
                start_cell = self._find_nearest_unblocked(start_cell, grid, grid_info)
                if start_cell is None:
                    return []

            if grid[goal_cell]["blocked"]:
                goal_cell = self._find_nearest_unblocked(goal_cell, grid, grid_info)
                if goal_cell is None:
                    return []

        # Initialize
        open_set = []
        closed_set = set()
        g_costs = defaultdict(lambda: float('inf'))
        g_costs[start_cell] = 0

        # Create start node
        h = self._heuristic(start_cell, goal_cell)
        start_node = PathNode(
            position=start_cell,
            g_cost=0,
            h_cost=h,
            f_cost=h
        )
        heapq.heappush(open_set, start_node)

        # Track parents for path reconstruction
        parents = {}

        while open_set:
            current = heapq.heappop(open_set)
            current_cell = current.position

            if current_cell in closed_set:
                continue

            closed_set.add(current_cell)

            # Check if reached goal
            if current_cell == goal_cell:
                return self._reconstruct_path(parents, current_cell, grid_info)

            # Expand neighbors
            neighbors = self._get_neighbors(current_cell, grid_info)

            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue

                if grid[neighbor]["blocked"]:
                    continue

                # Calculate costs
                move_cost = self._get_move_cost(current_cell, neighbor, grid)
                tentative_g = g_costs[current_cell] + move_cost

                if tentative_g < g_costs[neighbor]:
                    parents[neighbor] = current_cell
                    g_costs[neighbor] = tentative_g

                    h = self._heuristic(neighbor, goal_cell)
                    f = tentative_g + h * self._heuristic_weight

                    node = PathNode(
                        position=neighbor,
                        g_cost=tentative_g,
                        h_cost=h,
                        f_cost=f
                    )
                    heapq.heappush(open_set, node)

        # No path found
        return []

    def _find_nearest_unblocked(
        self,
        cell: Tuple[int, int],
        grid: Dict,
        grid_info: Dict,
        max_radius: int = 5
    ) -> Optional[Tuple[int, int]]:
        """Find nearest unblocked cell."""
        for r in range(1, max_radius + 1):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if abs(dr) == r or abs(dc) == r:
                        neighbor = (cell[0] + dr, cell[1] + dc)
                        if not grid[neighbor]["blocked"]:
                            return neighbor
        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate heuristic (estimated cost) between two cells."""
        if self._diagonal_movement:
            # Diagonal distance
            dx = abs(a[1] - b[1])
            dy = abs(a[0] - b[0])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            # Manhattan distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, cell: Tuple[int, int], grid_info: Dict) -> List[Tuple[int, int]]:
        """Get neighboring cells."""
        if self._diagonal_movement:
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
        else:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        neighbors = []
        for dr, dc in directions:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if (0 <= neighbor[0] < grid_info["rows"] and
                0 <= neighbor[1] < grid_info["cols"]):
                neighbors.append(neighbor)

        return neighbors

    def _get_move_cost(
        self,
        from_cell: Tuple[int, int],
        to_cell: Tuple[int, int],
        grid: Dict
    ) -> float:
        """Get movement cost between cells."""
        base_cost = grid[to_cell]["cost"]

        # Diagonal movement costs more
        if from_cell[0] != to_cell[0] and from_cell[1] != to_cell[1]:
            return base_cost * math.sqrt(2)

        return base_cost

    def _reconstruct_path(
        self,
        parents: Dict,
        goal_cell: Tuple[int, int],
        grid_info: Dict
    ) -> List[Tuple[float, float, float]]:
        """Reconstruct path from parents dictionary."""
        path = []
        current = goal_cell

        while current in parents:
            world_pos = self._cell_to_world(current, grid_info)
            path.append(world_pos)
            current = parents[current]

        # Add start position
        path.append(self._cell_to_world(current, grid_info))

        # Reverse to get start-to-goal order
        path.reverse()

        return path

    def _generate_alternatives(
        self,
        grid: Dict,
        grid_info: Dict,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        primary_path: List[Tuple[float, float, float]]
    ) -> List[List[Tuple[float, float, float]]]:
        """Generate alternative paths.

        This uses path diversification by modifying costs.
        """
        alternatives = []

        # Simple approach: try paths with different cost penalties
        for penalty in [2.0, 5.0]:
            # Create modified grid
            modified_grid = defaultdict(lambda: {"cost": 1.0, "blocked": False})
            for k, v in grid.items():
                modified_grid[k] = v.copy()

            # Penalize cells along primary path
            for pos in primary_path:
                cell = self._world_to_cell(pos, grid_info)
                modified_grid[cell]["cost"] *= penalty

            # Run A* with modified costs
            alt_path = self._astar_search(modified_grid, grid_info, start, goal)

            if alt_path and alt_path != primary_path:
                # Check if path is sufficiently different
                similarity = self._path_similarity(primary_path, alt_path)
                if similarity < 0.7:  # Less than 70% similar
                    alternatives.append(alt_path)

            if len(alternatives) >= self._max_alternatives:
                break

        return alternatives

    def _path_similarity(
        self,
        path1: List[Tuple[float, float, float]],
        path2: List[Tuple[float, float, float]]
    ) -> float:
        """Calculate similarity between two paths."""
        if not path1 or not path2:
            return 0.0

        # Sample points for comparison
        sample_size = min(len(path1), len(path2), 20)
        step1 = max(1, len(path1) // sample_size)
        step2 = max(1, len(path2) // sample_size)

        samples1 = path1[::step1][:sample_size]
        samples2 = path2[::step2][:sample_size]

        # Calculate average distance
        total_dist = 0.0
        for p1, p2 in zip(samples1, samples2):
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2)
            total_dist += dist

        avg_dist = total_dist / min(len(samples1), len(samples2))

        # Convert to similarity (0-1)
        # 5m average distance = 0 similarity
        similarity = max(0, 1 - avg_dist / 5.0)

        return similarity

    def _score_paths(
        self,
        paths: List[List[Tuple[float, float, float]]],
        blocked_positions: List[Tuple[float, float, float]],
        blocked_areas: List[Dict]
    ) -> List[float]:
        """Score paths based on safety and efficiency.

        Higher score = better path.
        """
        scores = []

        for path in paths:
            if not path:
                scores.append(0.0)
                continue

            # Base score from path length (shorter is better)
            length = self._calculate_path_length(path)
            length_score = max(0, 1.0 - length / 50.0)  # 50m = 0 score

            # Safety score (distance from blocked areas)
            min_distance = float('inf')
            for pos in path:
                for bp in blocked_positions:
                    dist = math.sqrt((pos[0] - bp[0]) ** 2 + (pos[2] - bp[2]) ** 2)
                    min_distance = min(min_distance, dist)

                for area in blocked_areas:
                    center = area.get("center", (0, 0, 0))
                    radius = area.get("radius", 1.0)
                    dist = math.sqrt((pos[0] - center[0]) ** 2 + (pos[2] - center[2]) ** 2)
                    min_distance = min(min_distance, dist - radius)

            safety_score = min(1.0, max(0, min_distance / 3.0))  # 3m = full score

            # Combined score
            score = 0.4 * length_score + 0.6 * safety_score
            scores.append(score)

        return scores

    def _calculate_path_length(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dz = path[i][2] - path[i-1][2]
            total += math.sqrt(dx * dx + dz * dz)

        return total

    def _check_path_safety(
        self,
        path: List,
        blocked_positions: List[Tuple[float, float, float]],
        blocked_areas: List[Dict],
        margin: float = 0.5
    ) -> bool:
        """Check if path avoids all blocked areas."""
        for pos in path:
            pos_tuple = tuple(pos) if isinstance(pos, list) else pos

            for bp in blocked_positions:
                dist = math.sqrt((pos_tuple[0] - bp[0]) ** 2 + (pos_tuple[2] - bp[2]) ** 2)
                if dist < margin:
                    return False

            for area in blocked_areas:
                center = area.get("center", (0, 0, 0))
                radius = area.get("radius", 1.0)
                dist = math.sqrt((pos_tuple[0] - center[0]) ** 2 + (pos_tuple[2] - center[2]) ** 2)
                if dist < radius + margin:
                    return False

        return True

    def get_last_result(self) -> Optional[PathResult]:
        """Get the most recent replanning result."""
        return self._last_result

    def path_to_actions(
        self,
        path: List[Tuple[float, float, float]],
        current_rotation: float = 0.0,
        step_size: float = 0.25,
        turn_angle: float = 15.0
    ) -> List[str]:
        """Convert path to action sequence.

        Args:
            path: List of waypoints
            current_rotation: Current facing angle in degrees
            step_size: Distance per forward action
            turn_angle: Degrees per turn action

        Returns:
            List of action strings
        """
        if len(path) < 2:
            return []

        actions = []
        current_angle = current_rotation

        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]

            # Calculate required angle
            dx = curr[0] - prev[0]
            dz = curr[2] - prev[2]
            target_angle = math.degrees(math.atan2(dx, -dz))
            if target_angle < 0:
                target_angle += 360

            # Calculate turn difference
            angle_diff = target_angle - current_angle
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360

            # Add turn actions
            turns_needed = int(abs(angle_diff) / turn_angle)
            if angle_diff > 0:
                actions.extend(["turn_left"] * turns_needed)
            else:
                actions.extend(["turn_right"] * turns_needed)

            # Calculate forward steps
            distance = math.sqrt(dx * dx + dz * dz)
            forward_steps = int(distance / step_size)
            actions.extend(["forward"] * max(1, forward_steps))

            current_angle = target_angle

        return actions