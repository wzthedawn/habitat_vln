#!/usr/bin/env python3
"""Simplified R2R evaluation without Habitat initialization delay."""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.r2r_loader import R2REpisode
from data.connectivity_graph import ConnectivityGraph


@dataclass
class SimpleResult:
    episode_id: int
    success: bool
    spl: float
    distance_to_goal: float
    steps: int
    instruction: str
    
    def to_dict(self):
        return asdict(self)


def run_simple_evaluation(num_episodes: int = 5, max_steps: int = 30, success_distance: float = 3.0):
    """Run simplified R2R evaluation."""
    print("=" * 60)
    print("R2R Simplified Evaluation (No Habitat)")
    print("=" * 60)
    
    # Create synthetic connectivity
    conn = ConnectivityGraph()
    points = [(float(x), 0.0, float(z)) for x in range(-10, 11, 2) for z in range(-10, 11, 2)]
    conn.build_from_navigable_points(points, distance_threshold=2.5)
    print(f"Created connectivity: {len(conn)} nodes")
    
    # Create mock episodes
    episodes = []
    vp_ids = list(conn.nodes.keys())
    
    for i in range(num_episodes):
        start_vp = random.choice(vp_ids)
        goal_vp = random.choice([v for v in vp_ids if v != start_vp])
        
        path = conn.shortest_path(start_vp, goal_vp) or [start_vp, goal_vp]
        trajectory = [[vp, 0.0, 0.0] for vp in path]
        
        episode = R2REpisode(
            episode_id=i,
            scene_id='test_scene',
            trajectory=trajectory,
            instructions=['Navigate to the goal', 'Find the target', 'Walk to destination'],
            start_position=conn.get_position(start_vp),
            goal_position=conn.get_position(goal_vp),
            geodesic_distance=conn.geodesic_distance(start_vp, goal_vp),
        )
        episodes.append(episode)
    
    print(f"Created {len(episodes)} episodes\n")
    
    # Run evaluation
    results = []
    for ep in episodes:
        print(f"Episode {ep.episode_id}:")
        print(f"  Start: {ep.start_position}")
        print(f"  Goal: {ep.goal_position}")
        print(f"  Geodesic: {ep.geodesic_distance:.2f}m")
        
        # Simulate navigation
        trajectory = [ep.start_position]
        current = list(ep.start_position)
        
        for step in range(max_steps):
            # Simple navigation: move toward goal with noise
            alpha = 0.15
            noise = [random.uniform(-0.2, 0.2) for _ in range(3)]
            current = [
                current[0] + (ep.goal_position[0] - current[0]) * alpha + noise[0],
                current[1] + (ep.goal_position[1] - current[1]) * alpha + noise[1],
                current[2] + (ep.goal_position[2] - current[2]) * alpha + noise[2],
            ]
            trajectory.append(tuple(current))
            
            # Check if reached goal
            dist = ((current[0] - ep.goal_position[0])**2 + 
                    (current[2] - ep.goal_position[2])**2)**0.5
            if dist < success_distance:
                break
        
        # Calculate metrics
        final_dist = ((trajectory[-1][0] - ep.goal_position[0])**2 + 
                      (trajectory[-1][2] - ep.goal_position[2])**2)**0.5
        success = final_dist < success_distance
        
        # Calculate trajectory length
        traj_length = sum(
            ((trajectory[i][0] - trajectory[i-1][0])**2 + 
             (trajectory[i][2] - trajectory[i-1][2])**2)**0.5
            for i in range(1, len(trajectory))
        )
        
        # SPL
        spl = ep.geodesic_distance / traj_length if success and traj_length > 0 else 0
        
        result = SimpleResult(
            episode_id=ep.episode_id,
            success=success,
            spl=spl,
            distance_to_goal=final_dist,
            steps=len(trajectory) - 1,
            instruction=ep.get_instruction(),
        )
        results.append(result)
        
        print(f"  Steps: {result.steps}")
        print(f"  Final distance: {final_dist:.2f}m")
        print(f"  Success: {success}")
        print(f"  SPL: {spl:.3f}")
        print()
    
    # Summary
    total = len(results)
    successes = sum(1 for r in results if r.success)
    avg_spl = sum(r.spl for r in results) / total if total > 0 else 0
    avg_dist = sum(r.distance_to_goal for r in results) / total if total > 0 else 0
    avg_steps = sum(r.steps for r in results) / total if total > 0 else 0
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Episodes: {total}")
    print(f"Success Rate: {successes/total*100:.1f}%")
    print(f"Avg SPL: {avg_spl:.3f}")
    print(f"Avg Distance to Goal: {avg_dist:.2f}m")
    print(f"Avg Steps: {avg_steps:.1f}")
    print("=" * 60)
    
    # Save results
    output = {
        "summary": {
            "num_episodes": total,
            "success_rate": successes / total,
            "spl": avg_spl,
            "avg_distance_to_goal": avg_dist,
            "avg_steps": avg_steps,
        },
        "episodes": [r.to_dict() for r in results],
        "timestamp": datetime.now().isoformat(),
    }
    
    with open("results_r2r_simple.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: results_r2r_simple.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--success-distance", type=float, default=3.0)
    args = parser.parse_args()
    
    run_simple_evaluation(args.episodes, args.max_steps, args.success_distance)
