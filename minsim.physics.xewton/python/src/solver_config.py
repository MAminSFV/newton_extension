"""Solver configuration for the Xewton custom solver.

Adds a single new solver type, ``"xsolver"``, on top of the upstream
``isaacsim.physics.newton`` solver-config taxonomy. :class:`XSolver` is the
runtime implementation in :mod:`.xsolver`; this file is the data-only config
the rest of the stack consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from isaacsim.physics.newton.impl.solver_config import NewtonSolverConfig


@dataclass
class XSolverConfig(NewtonSolverConfig):
    """Config for the demonstration :class:`XSolver`.

    XSolver wraps a delegate integrator (default: SolverSemiImplicit) and
    doubles the +X linear component of every rigid-body wrench before
    delegating. Matches the contract of ``newton.solvers.SolverBase``.
    """

    solver_type: Literal["xsolver"] = "xsolver"
    """Solver discriminator; consumed by :meth:`XewtonStage._get_solver`."""
