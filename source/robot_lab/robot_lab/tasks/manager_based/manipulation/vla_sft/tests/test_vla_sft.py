#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify VLA-SFT Layer 1 and Layer 2 imports and basic functionality.

This script tests:
1. Scene registry initialization (Basic and Mobile)
2. Scene sampling
3. Instruction generation
4. Configuration loading
5. Scene manager multi-layer support

Usage:
    python -m robot_lab.tasks.manager_based.manipulation.vla_sft.tests.test_vla_sft
"""

import sys
from pathlib import Path


def test_scene_registry():
    """Test BasicGraspSceneRegistry."""
    print("Testing BasicGraspSceneRegistry...")

    from ..scenes import BasicGraspSceneRegistry

    registry = BasicGraspSceneRegistry(seed=42)

    # Test scene counts
    counts = registry.get_scene_counts()
    print(f"  Scene counts: {counts}")
    assert sum(counts.values()) == 40, f"Expected 40 scenes, got {sum(counts.values())}"

    # Test listing scenes
    all_scenes = registry.list_scenes()
    print(f"  Total scenes: {len(all_scenes)}")
    assert len(all_scenes) == 40

    # Test sampling
    scene = registry.sample_scene()
    print(f"  Sampled scene: {scene.scene_id}, type: {scene.scene_type}")

    # Test getting specific scene
    scene = registry.get_scene("a1_ground_grasp_005")
    print(f"  Got scene: {scene.scene_id}")

    print("  ✓ BasicGraspSceneRegistry tests passed!")


def test_mobile_scene_registry():
    """Test MobileGraspSceneRegistry."""
    print("\nTesting MobileGraspSceneRegistry...")

    from ..scenes import MobileGraspSceneRegistry

    registry = MobileGraspSceneRegistry(seed=42)

    # Test scene counts
    counts = registry.get_scene_counts()
    print(f"  Scene counts: {counts}")
    assert sum(counts.values()) == 40, f"Expected 40 scenes, got {sum(counts.values())}"

    # Test listing scenes
    all_scenes = registry.list_scenes()
    print(f"  Total scenes: {len(all_scenes)}")
    assert len(all_scenes) == 40

    # Test sampling by type
    b1_scene = registry.sample_scene("b1_open_floor")
    print(f"  Sampled B1 scene: {b1_scene.scene_id}, type: {b1_scene.scene_type}")
    assert b1_scene.scene_type == "b1_open_floor"

    # Test getting specific scene
    b2_scene = registry.get_scene("b2_table_approach_003")
    print(f"  Got B2 scene: {b2_scene.scene_id}")
    assert b2_scene.scene_type == "b2_table_approach"

    print("  ✓ MobileGraspSceneRegistry tests passed!")


def test_scene_config():
    """Test BasicGraspSceneConfig."""
    print("\nTesting BasicGraspSceneConfig...")

    from ..scenes import BasicGraspSceneA1

    scene = BasicGraspSceneA1()

    # Test pose sampling
    pos, quat = scene.sample_object_pose()
    print(f"  Sampled position: {pos}")
    print(f"  Sampled quaternion: {quat}")
    assert pos.shape == (3,)
    assert quat.shape == (4,)

    # Test color sampling
    color = scene.sample_object_color()
    print(f"  Sampled color: {color}")
    assert len(color) == 3

    # Test object type sampling
    obj_type = scene.sample_object_type()
    print(f"  Sampled object type: {obj_type}")
    assert obj_type in scene.object_types

    # Test instruction generation
    instruction = scene.generate_instruction("cube")
    print(f"  Generated instruction: {instruction}")
    assert "cube" in instruction.lower() or "block" in instruction.lower()

    print("  ✓ BasicGraspSceneConfig tests passed!")


def test_mobile_scene_config():
    """Test MobileGraspSceneConfig."""
    print("\nTesting MobileGraspSceneConfig...")

    from ..scenes import MobileGraspSceneB1, MobileGraspSceneB3, MobileGraspSceneB4

    # Test B1: Open floor
    b1_scene = MobileGraspSceneB1()
    pos, angle, distance = b1_scene.sample_target_pose()
    print(f"  B1 target pos: {pos}, angle: {angle:.3f}, distance: {distance:.3f}")
    assert 1.5 <= distance <= 3.0, f"Expected distance 1.5-3.0, got {distance}"

    base_pos, base_yaw = b1_scene.sample_base_init_pose()
    print(f"  B1 base pos: {base_pos}, yaw: {base_yaw:.3f}")
    assert base_pos.shape == (2,)

    instruction = b1_scene.generate_instruction("cube")
    print(f"  B1 instruction: {instruction}")
    assert "navigate" in instruction.lower() or "go" in instruction.lower()

    # Test B3: Obstacle avoidance
    b3_scene = MobileGraspSceneB3()
    target_pos = b3_scene.sample_target_pose()[0]
    obstacles = b3_scene.sample_obstacles(target_pos)
    print(f"  B3 obstacles: {len(obstacles)}")
    assert 2 <= len(obstacles) <= 4, f"Expected 2-4 obstacles, got {len(obstacles)}"

    # Test B4: Partial occlusion
    b4_scene = MobileGraspSceneB4()
    target_pos = b4_scene.sample_target_pose()[0]
    occluder = b4_scene.sample_occluder(target_pos)
    print(f"  B4 occluder: type={occluder['type']}, pos={occluder['position']}")
    assert occluder["type"] == "tall_block"

    print("  ✓ MobileGraspSceneConfig tests passed!")


def test_instruction_generator():
    """Test InstructionGenerator."""
    print("\nTesting InstructionGenerator...")

    from ..data_collection import InstructionGenerator
    from ..scenes import BasicGraspSceneA2, MobileGraspSceneB1

    gen = InstructionGenerator(seed=42)

    # Test basic generation
    instruction = gen.generate("basic_grasp", {"object": "cube"})
    print(f"  Generated instruction: {instruction}")
    assert "cube" in instruction.lower() or "block" in instruction.lower() or "box" in instruction.lower()

    # Test scene type mapping
    scene_a2 = BasicGraspSceneA2()
    instruction = gen.generate_for_scene_config(scene_a2)
    print(f"  A2 instruction: {instruction}")
    assert "table" in instruction.lower()

    print("  ✓ InstructionGenerator tests passed!")


def test_scene_manager():
    """Test VLASSceneManager with multi-layer support."""
    print("\nTesting VLASSceneManager...")

    from ..data_collection import VLASSceneManager

    manager = VLASSceneManager(seed=42)

    # Test total scene count
    total = manager.total_scenes
    print(f"  Total scenes (all layers): {total}")
    assert total == 80, f"Expected 80 scenes (40+40), got {total}"

    # Test scene counts by type
    counts = manager.get_scene_counts()
    print(f"  Scene counts by type: {counts}")
    assert "a1_ground_grasp" in counts
    assert "b1_open_floor" in counts

    # Test sampling from basic layer
    basic_scene = manager.sample_scene(layer="basic")
    print(f"  Sampled basic scene: {basic_scene.scene_id}, type: {basic_scene.scene_type}")

    # Test sampling from mobile layer
    mobile_scene = manager.sample_scene(layer="mobile")
    print(f"  Sampled mobile scene: {mobile_scene.scene_id}, type: {mobile_scene.scene_type}")

    # Test listing all scenes
    all_scenes = manager.list_scenes()
    print(f"  All scenes: {len(all_scenes)}")
    assert len(all_scenes) == 80

    # Test listing by layer
    basic_scenes = manager.list_scenes(layer="basic")
    mobile_scenes = manager.list_scenes(layer="mobile")
    print(f"  Basic scenes: {len(basic_scenes)}, Mobile scenes: {len(mobile_scenes)}")
    assert len(basic_scenes) == 40
    assert len(mobile_scenes) == 40

    # Test getting specific scene by ID
    scene = manager.get_scene_by_id("b3_obstacle_avoidance_007")
    print(f"  Got scene by ID: {scene.scene_id}, type: {scene.scene_type}")

    # Test randomization params for mobile scene
    params = manager.get_randomization_params(mobile_scene)
    print(f"  Mobile randomization keys: {list(params.keys())}")
    assert "target_position" in params
    assert "base_init_position" in params

    print("  ✓ VLASSceneManager tests passed!")


def test_config_imports():
    """Test configuration imports."""
    print("\nTesting configuration imports...")

    try:
        from ..configs import (
            Go2X5VLASBasicEnvCfg,
            Go2X5VLASBasicEnvCfg_PLAY,
            Go2X5VLASMobileEnvCfg,
            Go2X5VLASMobileEnvCfg_PLAY,
        )
        print("  Configuration classes imported successfully")
        print("    - Go2X5VLASBasicEnvCfg")
        print("    - Go2X5VLASMobileEnvCfg")

        # Note: We don't instantiate the full config here as it requires
        # Isaac Lab to be fully initialized

        print("  ✓ Configuration imports passed!")
    except Exception as e:
        print(f"  ⚠ Configuration import skipped (Isaac Lab may not be available): {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("VLA-SFT Tests (Layer 1 & 2)")
    print("=" * 60)

    try:
        test_scene_registry()
        test_mobile_scene_registry()
        test_scene_config()
        test_mobile_scene_config()
        test_instruction_generator()
        test_scene_manager()
        test_config_imports()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
