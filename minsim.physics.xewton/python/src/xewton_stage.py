"""Xewton simulation stage.

Subclass of :class:`isaacsim.physics.newton.NewtonStage` that wires up the
custom :class:`XSolver` when the active solver_cfg has
``solver_type == "xsolver"``. All USD parsing, fabric sync, timeline
handling and the simulate loop are inherited unchanged.
"""

from __future__ import annotations

import newton

from isaacsim.physics.newton.impl.newton_stage import NewtonStage

from .solver_config import XSolverConfig
from .xsolver import XSolver


class XewtonStage(NewtonStage):
    """NewtonStage that knows how to construct the XSolver custom solver."""

    @classmethod
    def _get_solver(cls, model: newton.Model, solver_cfg):
        """Construct the solver for this step.

        Falls through to :meth:`NewtonStage._get_solver` for stock solver
        types (xpbd, mujoco). Adds an ``"xsolver"`` branch that wraps a
        ``SolverSemiImplicit`` delegate inside :class:`XSolver`.
        """
        if isinstance(solver_cfg, XSolverConfig) or getattr(solver_cfg, "solver_type", None) == "xsolver":
            delegate = newton.solvers.SolverSemiImplicit(model)
            return XSolver(model, delegate=delegate)
        return NewtonStage._get_solver(model, solver_cfg)
