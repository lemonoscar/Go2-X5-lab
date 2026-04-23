# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Instruction generation for VLA-SFT data collection.

This module provides utilities for generating natural language
instructions compatible with the VLA format used in SimpleVLA-RL.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


class InstructionGenerator:
    """Generate natural language instructions for VLA data collection.

    This class manages instruction templates and generates diverse
    instructions by sampling from templates and replacing placeholders.

    Example:
        >>> gen = InstructionGenerator()
        >>> instruction = gen.generate("basic_grasp", {"object": "cube"})
        >>> "pick up the cube from the ground"
    """

    # Instruction templates by scene type
    TEMPLATES: Dict[str, List[str]] = {
        "basic_grasp": [
            "pick up the {object} from the ground",
            "grasp the {object}",
            "collect the {object} from the floor",
            "pick up the {object}",
        ],
        "table_grasp": [
            "pick up the {object} from the table",
            "grasp the {object} on the table",
            "take the {object} from the table",
            "retrieve the {object} from the table",
        ],
        "clutter_grasp": [
            "pick up the {object} among the other objects",
            "grasp the {object} from the cluttered table",
            "find and collect the {object}",
            "pick up the {object} in the clutter",
        ],
    }

    # Object descriptions mapping
    OBJECT_DESCRIPTIONS: Dict[str, Tuple[List[str], List[str]]] = {
        "cube": (
            ["cube", "block", "square block", "box"],
            ["hexagonal prism", "octahedron"],
        ),
        "sphere": (
            ["ball", "sphere", "round object"],
            ["ellipsoid", "hemisphere"],
        ),
        "cylinder": (
            ["cylinder", "can", "cylindrical object"],
            ["tube", "rounded column"],
        ),
        "cup": (
            ["cup", "mug", "small container"],
            ["goblet", "teacup"],
        ),
        "bowl": (
            ["bowl", "dish", "round container"],
            ["basin", "soup bowl"],
        ),
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the instruction generator.

        Args:
            seed: Random seed for reproducible generation.
        """
        self._rng = random.Random(seed)

    def generate(
        self,
        scene_type: str,
        params: Dict[str, str],
        use_unseen_description: bool = False,
    ) -> str:
        """Generate an instruction for a given scene type.

        Args:
            scene_type: Type of scene (e.g., "basic_grasp", "table_grasp").
            params: Parameter dict with placeholder values (e.g., {"object": "cube"}).
            use_unseen_description: Whether to use unseen object descriptions.

        Returns:
            Generated instruction string.
        """
        # Get templates for scene type
        if scene_type not in self.TEMPLATES:
            # Default to basic_grasp templates
            templates = self.TEMPLATES["basic_grasp"]
        else:
            templates = self.TEMPLATES[scene_type]

        # Sample a template
        template = self._rng.choice(templates)

        # Replace placeholders
        instruction = self._replace_placeholders(
            template,
            params,
            use_unseen_description=use_unseen_description,
        )

        return instruction

    def _replace_placeholders(
        self,
        template: str,
        params: Dict[str, str],
        use_unseen_description: bool = False,
    ) -> str:
        """Replace placeholders in template with values.

        Args:
            template: Instruction template with {placeholders}.
            params: Parameter dict.
            use_unseen_description: Whether to use unseen descriptions.

        Returns:
            Template with placeholders replaced.
        """
        result = template
        for key, value in params.items():
            placeholder = "{" + key + "}"

            # Check if value is an object type that needs description expansion
            if key == "object" and value in self.OBJECT_DESCRIPTIONS:
                seen_descs, unseen_descs = self.OBJECT_DESCRIPTIONS[value]
                if use_unseen_description and unseen_descs:
                    descriptions = unseen_descs
                else:
                    descriptions = seen_descs
                value = self._rng.choice(descriptions)

            result = result.replace(placeholder, value)

        return result

    def generate_for_scene_config(
        self,
        scene_config,
        object_type: Optional[str] = None,
    ) -> str:
        """Generate instruction based on scene configuration.

        Args:
            scene_config: Scene configuration object (from BasicGraspSceneConfig).
            object_type: Override object type (uses sampled if None).

        Returns:
            Generated instruction string.
        """
        if object_type is None:
            object_type = scene_config.sample_object_type(self._rng)

        # Determine scene type from config
        scene_type = getattr(scene_config, "scene_type", "basic_grasp")

        # Map scene_type to instruction template type
        template_type = "basic_grasp"
        if "table" in scene_type:
            template_type = "table_grasp"
        elif "clutter" in scene_type:
            template_type = "clutter_grasp"

        return self.generate(template_type, {"object": object_type})

    def get_all_templates(self) -> Dict[str, List[str]]:
        """Get all instruction templates.

        Returns:
            Dictionary mapping scene types to template lists.
        """
        return self.TEMPLATES.copy()

    def add_template(self, scene_type: str, template: str):
        """Add a new instruction template.

        Args:
            scene_type: Scene type for the template.
            template: Template string with {placeholders}.
        """
        if scene_type not in self.TEMPLATES:
            self.TEMPLATES[scene_type] = []
        self.TEMPLATES[scene_type].append(template)


def generate_episode_instructions(
    scene_type: str,
    object_type: str,
    num_instructions: int = 10,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate multiple instruction variations for an episode.

    This is useful for data augmentation where you want multiple
    instruction variations for the same trajectory.

    Args:
        scene_type: Type of scene.
        object_type: Type of target object.
        num_instructions: Number of instruction variations to generate.
        seed: Random seed.

    Returns:
        List of generated instructions.
    """
    gen = InstructionGenerator(seed=seed)
    params = {"object": object_type}

    instructions = []
    for _ in range(num_instructions):
        instruction = gen.generate(scene_type, params)
        instructions.append(instruction)

    return instructions
