# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
# import toml  # 未使用，注释掉

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", "vla_sft"]  # vla_sft 有版本兼容问题，暂时跳过
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
