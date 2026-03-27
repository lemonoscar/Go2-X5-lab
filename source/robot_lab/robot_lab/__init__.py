# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
try:  # pragma: no cover - optional in headless or smoke-only environments
    from .ui_extension_example import *
except Exception:
    pass
