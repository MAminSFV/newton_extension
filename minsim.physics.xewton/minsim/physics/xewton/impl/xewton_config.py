"""Xewton simulation config.

Thin subclass of :class:`isaacsim.physics.newton.NewtonConfig` whose only
purpose is to default the solver to :class:`XSolverConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isaacsim.physics.newton.impl.newton_config import NewtonConfig
from isaacsim.physics.newton.impl.solver_config import NewtonSolverConfig

from .solver_config import XSolverConfig


@dataclass
class XewtonConfig(NewtonConfig):
    """NewtonConfig variant whose default solver is :class:`XSolver`."""

    solver_cfg: NewtonSolverConfig = field(default_factory=XSolverConfig)
