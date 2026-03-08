#!/usr/bin/env python3
"""Download R2R dataset for VLN evaluation."""

import argparse
import os
import sys
from pathlib import Path
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("download_r2r")


# R2R dataset URLs
R2R_URLS = {
    "train": "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/tasks/R2R/data/R2R_train.json",
    "val_seen": "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/tasks/R2R/data/R2R_val_seen.json",
    "val_unseen": "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/tasks/R2R/data/R2R_val_unseen.json",
    "test": "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/tasks/R2R/data/R2R_test.json",
}

# Connectivity base URL
CONNECTIVITY_BASE_URL = "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/connectivity/"


def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from URL.

    Args:
        url: URL to download
        output_path: Local path to save

    Returns:
        True if successful
    """
    try:
        import urllib.request

        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_r2r_data(output_dir: str, splits: list = None) -> dict:
    """
    Download R2R dataset.

    Args:
        output_dir: Output directory
        splits: List of splits to download (default: all)

    Returns:
        Download statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = list(R2R_URLS.keys())

    stats = {"success": [], "failed": []}

    for split in splits:
        if split not in R2R_URLS:
            logger.warning(f"Unknown split: {split}")
            continue

        url = R2R_URLS[split]
        output_file = output_path / f"R2R_{split}.json"

        if output_file.exists():
            logger.info(f"File already exists: {output_file}")
            stats["success"].append(split)
            continue

        if download_file(url, str(output_file)):
            stats["success"].append(split)
        else:
            stats["failed"].append(split)

    return stats


def download_connectivity(output_dir: str, scene_ids: list = None) -> dict:
    """
    Download connectivity files for scenes.

    Args:
        output_dir: Output directory
        scene_ids: List of scene IDs (default: will extract from R2R data)

    Returns:
        Download statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0, "skipped": 0}

    # If no scene IDs provided, try to extract from R2R data
    if scene_ids is None:
        scene_ids = get_scene_ids_from_r2r(output_path.parent / "R2R")

    if not scene_ids:
        logger.warning("No scene IDs to download connectivity for")
        return stats

    logger.info(f"Downloading connectivity for {len(scene_ids)} scenes...")

    for scene_id in scene_ids:
        url = f"{CONNECTIVITY_BASE_URL}{scene_id}_connectivity.json"
        output_file = output_path / f"{scene_id}_connectivity.json"

        if output_file.exists():
            logger.info(f"File already exists: {output_file}")
            stats["skipped"] += 1
            continue

        if download_file(url, str(output_file)):
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


def get_scene_ids_from_r2r(r2r_dir: str) -> list:
    """
    Extract unique scene IDs from R2R data.

    Args:
        r2r_dir: Directory containing R2R JSON files

    Returns:
        List of unique scene IDs
    """
    r2r_path = Path(r2r_dir)
    scene_ids = set()

    for json_file in r2r_path.glob("R2R_*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            for item in data:
                scan = item.get("scan", "")
                if scan:
                    scene_ids.add(scan)

        except Exception as e:
            logger.warning(f"Could not read {json_file}: {e}")

    return list(scene_ids)


def create_sample_data(output_dir: str) -> None:
    """
    Create sample R2R-style data for testing.

    This creates mock data when real R2R data is not available.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sample episodes
    sample_episodes = []
    for i in range(10):
        episode = {
            "episode_id": i,
            "instruction_id": i,
            "trajectory": [
                [f"vp_{j}", 0.0, 0.0]
                for j in range(5)
            ],
            "instructions": [
                f"Sample instruction 1 for episode {i}",
                f"Sample instruction 2 for episode {i}",
                f"Sample instruction 3 for episode {i}",
            ],
            "path_id": i,
            "scan": "sample_scene",
            "heading": 0.0,
        }
        sample_episodes.append(episode)

    # Save sample data
    sample_file = output_path / "R2R_sample.json"
    with open(sample_file, "w") as f:
        json.dump(sample_episodes, f, indent=2)

    logger.info(f"Created sample data: {sample_file}")

    # Create sample connectivity
    connectivity = {}
    for i in range(20):
        vp_id = f"vp_{i}"
        connectivity[vp_id] = {
            "image_id": vp_id,
            "position": [float(i), 0.0, float(i)],
            "unobstructed": [f"vp_{j}" for j in range(max(0, i-2), min(20, i+3)) if j != i],
        }

    conn_file = output_path.parent / "connectivity" / "sample_scene_connectivity.json"
    conn_file.parent.mkdir(parents=True, exist_ok=True)
    with open(conn_file, "w") as f:
        json.dump(list(connectivity.values()), f, indent=2)

    logger.info(f"Created sample connectivity: {conn_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download R2R dataset")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val_seen", "val_unseen", "test"],
        help="R2R splits to download",
    )
    parser.add_argument(
        "--connectivity",
        action="store_true",
        help="Also download connectivity files",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create sample data instead of downloading",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List scene IDs from downloaded R2R data",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.sample:
        logger.info("Creating sample data...")
        create_sample_data(str(output_dir / "R2R"))
        return

    if args.list_scenes:
        logger.info("Extracting scene IDs from R2R data...")
        scene_ids = get_scene_ids_from_r2r(str(output_dir / "R2R"))
        print(f"\nFound {len(scene_ids)} unique scenes:")
        for scene_id in sorted(scene_ids):
            print(f"  {scene_id}")
        return

    # Download R2R data
    logger.info("=" * 60)
    logger.info("Downloading R2R dataset")
    logger.info("=" * 60)

    r2r_stats = download_r2r_data(str(output_dir / "R2R"), args.splits)

    print(f"\nR2R Download Results:")
    print(f"  Success: {r2r_stats['success']}")
    print(f"  Failed: {r2r_stats['failed']}")

    # Download connectivity if requested
    if args.connectivity:
        logger.info("\n" + "=" * 60)
        logger.info("Downloading connectivity files")
        logger.info("=" * 60)

        conn_stats = download_connectivity(str(output_dir / "connectivity"))

        print(f"\nConnectivity Download Results:")
        print(f"  Success: {conn_stats['success']}")
        print(f"  Failed: {conn_stats['failed']}")
        print(f"  Skipped: {conn_stats['skipped']}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nData saved to: {output_dir}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/R2R/          - R2R episode data")
    print(f"  {output_dir}/connectivity/ - Scene connectivity graphs")


if __name__ == "__main__":
    main()