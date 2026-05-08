"""Xewton simulation stage.

Subclass of :class:`isaacsim.physics.newton.NewtonStage` that wires up the
custom :class:`XSolver` when the active solver_cfg has
``solver_type == "xsolver"``. All USD parsing, fabric sync, timeline
handling and the simulate loop are inherited unchanged.
"""

from __future__ import annotations

import carb
import newton

from isaacsim.physics.newton.impl.newton_stage import NewtonStage

from .solver_config import XSolverConfig
from .xsolver import XSolver


class XewtonStage(NewtonStage):
    """NewtonStage that knows how to construct the XSolver custom solver."""

    def __setattr__(self, name, value):
        """Intercept self.builder assignment to auto-register MuJoCo attributes.

        NewtonStage.initialize_newton always includes SchemaResolverMjc in the
        add_usd() call, which requires SolverMuJoCo.register_custom_attributes()
        to have run on the builder first — even for non-MuJoCo solvers.
        We hook the assignment of self.builder so registration happens on the
        fresh ModelBuilder before add_usd() is called, without needing to
        duplicate any of NewtonStage's initialization code.
        """
        super().__setattr__(name, value)
        if name == "builder" and value is not None:
            has_mujoco = any(
                attr.namespace == "mujoco"
                for attr in value.custom_attributes.values()
            )
            if not has_mujoco:
                carb.log_warn("[xewton] registering MuJoCo custom attributes on builder")
                newton.solvers.SolverMuJoCo.register_custom_attributes(value)

    @classmethod
    def _get_solver(cls, model: newton.Model, solver_cfg):
        if isinstance(solver_cfg, XSolverConfig) or getattr(solver_cfg, "solver_type", None) == "xsolver":
            delegate = newton.solvers.SolverSemiImplicit(model)
            return XSolver(model, delegate=delegate)
        return NewtonStage._get_solver(model, solver_cfg)
