"""Xewton physics simulation extension for Isaac Sim.

Mirrors :mod:`isaacsim.physics.newton.impl.extension` but wires up the
Xewton variant: :class:`XewtonConfig`, :class:`XewtonStage` (with the
custom :class:`XSolver`), and :class:`XewtonSimulationRegistry`
which registers under the engine name ``"Xewton"`` so it coexists with
the upstream ``"Newton"`` backend.
"""

from __future__ import annotations

import carb
import omni.ext
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.physics.newton.impl.interface import NewtonPhysicsInterface
from omni.physics.core import get_physics_interface, k_invalid_simulation_id

from .xewton_config import XewtonConfig
from .xewton_register import XewtonSimulationRegistry
from .xewton_stage import XewtonStage

# Engine name used for SimulationManager.switch_physics_engine and log tags.
ENGINE_NAME = "xewton"
EXT_NAME = "minsim.physics.xewton"

# Module-level singletons exposed via the public acquire_* helpers.
_xewton_physics_interface: NewtonPhysicsInterface | None = None
_xewton_stage: XewtonStage | None = None


def acquire_physics_interface() -> NewtonPhysicsInterface | None:
    """Return the Xewton physics interface (or None if not initialized)."""
    return _xewton_physics_interface


def acquire_stage() -> XewtonStage | None:
    """Return the Xewton simulation stage (or None if not initialized)."""
    return _xewton_stage


def get_active_physics_engine() -> str:
    """Name of the currently active physics engine, or ``"Unknown"``."""
    try:
        physics = get_physics_interface()
        if not physics:
            return "Unknown"
        for sim_id in physics.get_simulation_ids():
            if physics.is_simulation_active(sim_id):
                return physics.get_simulation_name(sim_id)
        return "Unknown"
    except Exception:
        return "Unknown"


def get_available_physics_engines(verbose: bool = False) -> list[tuple[str, bool]]:
    """List of ``(engine_name, is_active)`` tuples for all registered engines."""
    try:
        physics = get_physics_interface()
        if not physics:
            return []
        engines = [
            (physics.get_simulation_name(sim_id), physics.is_simulation_active(sim_id))
            for sim_id in physics.get_simulation_ids()
        ]
        if verbose:
            print("Available physics engines:")
            for name, active in engines:
                print(f"  {name}: {'active' if active else 'inactive'}")
            print("-" * 60)
        return engines
    except Exception:
        return []


class XewtonSimExtension(omni.ext.IExt):
    """Kit extension entry point for Xewton."""

    def on_startup(self, ext_id: str):
        global _xewton_stage, _xewton_physics_interface

        cfg = XewtonConfig()
        _xewton_stage = XewtonStage(cfg=cfg)
        _xewton_physics_interface = NewtonPhysicsInterface(_xewton_stage)

        self._xewton_registry = XewtonSimulationRegistry()
        simulation_id = self._xewton_registry.register_xewton(_xewton_stage)

        if simulation_id is None or simulation_id == k_invalid_simulation_id:
            carb.log_error(
                f"[{EXT_NAME}] Failed to register Xewton (solver: {cfg.solver_cfg.solver_type})"
            )
            self._auto_switched = False
            return

        carb.log_info(
            f"[{EXT_NAME}] Xewton registered with unified physics interface "
            f"(solver: {cfg.solver_cfg.solver_type})"
        )

        settings = carb.settings.get_settings()
        auto_switch = settings.get(f"/exts/{EXT_NAME}/auto_switch_on_startup")
        if auto_switch is True:
            success = SimulationManager.switch_physics_engine(ENGINE_NAME)
            self._auto_switched = bool(success)
            if success:
                carb.log_warn(f"[{EXT_NAME}] Auto-switched to {ENGINE_NAME} on startup")
            else:
                carb.log_error(f"[{EXT_NAME}] Failed to auto-switch to {ENGINE_NAME}")
        else:
            self._auto_switched = False
            carb.log_warn(
                f"[{EXT_NAME}] {ENGINE_NAME} registered but not auto-activated "
                f"(auto_switch_on_startup=false)"
            )

    def on_shutdown(self):
        global _xewton_stage

        if getattr(self, "_auto_switched", False):
            success = SimulationManager.switch_physics_engine("physx")
            if success:
                carb.log_warn(f"[{EXT_NAME}] Switched back to physx on shutdown")
            else:
                carb.log_warn(f"[{EXT_NAME}] Failed to switch back to physx on shutdown")

        if hasattr(self, "_xewton_registry"):
            self._xewton_registry.unregister_xewton()

        if _xewton_stage is not None:
            _xewton_stage.init()
