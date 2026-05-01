"""Xewton: a custom Newton solver demonstration extension for Isaac Sim.

Public API mirrors ``isaacsim.physics.newton`` and re-exports the Xewton
variants of the Config / Stage / Solver-config classes.
"""

from .impl.extension import (
    XewtonSimExtension,
    acquire_physics_interface,
    acquire_stage,
    get_active_physics_engine,
    get_available_physics_engines,
)
from .impl.solver_config import XSolverConfig
from .impl.xewton_config import XewtonConfig
from .impl.xewton_stage import XewtonStage
from .impl.xsolver import XSolver

__all__ = [
    "XewtonSimExtension",
    "acquire_physics_interface",
    "acquire_stage",
    "get_active_physics_engine",
    "get_available_physics_engines",
    "XewtonConfig",
    "XewtonStage",
    "XSolver",
    "XSolverConfig",
]
