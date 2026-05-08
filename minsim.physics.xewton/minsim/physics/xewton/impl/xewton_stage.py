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

    def initialize_newton(self, device):
        """Initialize Newton, ensuring MuJoCo attributes are always registered.

        SchemaResolverMjc is always included in add_usd() by NewtonStage, but
        SolverMuJoCo.register_custom_attributes() is only called when
        solver_type == "mujoco". For "xsolver" we briefly present as "mujoco"
        so the registration runs, then restore the original type and swap in
        the real XSolver.
        """
        solver_cfg = self.cfg.solver_cfg
        original_type = getattr(solver_cfg, "solver_type", None)
        needs_patch = original_type not in ("mujoco", "xpbd", None)

        if needs_patch:
            solver_cfg.solver_type = "mujoco"
        try:
            super().initialize_newton(device)
        finally:
            if needs_patch:
                solver_cfg.solver_type = original_type

        if needs_patch and self.model is not None:
            self.solver = type(self)._get_solver(self.model, solver_cfg)

    @classmethod
    def _get_solver(cls, model: newton.Model, solver_cfg):
        if isinstance(solver_cfg, XSolverConfig) or getattr(solver_cfg, "solver_type", None) == "xsolver":
            delegate = newton.solvers.SolverSemiImplicit(model)
            return XSolver(model, delegate=delegate)
        return NewtonStage._get_solver(model, solver_cfg)
