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
        """Intercept key attribute assignments to handle Newton API differences.

        1. self.builder — auto-register MuJoCo attributes before add_usd().
           NewtonStage always passes SchemaResolverMjc to add_usd(), which
           requires SolverMuJoCo.register_custom_attributes() even for
           non-MuJoCo solvers.

        2. self.model — wrap model.collide() to accept and silently drop
           rigid_contact_margin / soft_contact_margin kwargs absent in older
           Newton versions, avoiding a TypeError in NewtonStage.simulate().
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
        elif name == "model" and value is not None:
            import inspect
            _orig_collide = value.collide
            try:
                supports_margins = "rigid_contact_margin" in inspect.signature(_orig_collide).parameters
            except (ValueError, TypeError):
                supports_margins = True
            if not supports_margins:
                carb.log_warn("[xewton] wrapping model.collide() for older Newton API compatibility")
                def _collide_compat(state, rigid_contact_margin=None, soft_contact_margin=None, **kwargs):
                    return _orig_collide(state, **kwargs)
                value.collide = _collide_compat

    @classmethod
    def _get_solver(cls, model: newton.Model, solver_cfg):
        if isinstance(solver_cfg, XSolverConfig) or getattr(solver_cfg, "solver_type", None) == "xsolver":
            delegate = newton.solvers.SolverSemiImplicit(model)
            return XSolver(model, delegate=delegate)
        return NewtonStage._get_solver(model, solver_cfg)
