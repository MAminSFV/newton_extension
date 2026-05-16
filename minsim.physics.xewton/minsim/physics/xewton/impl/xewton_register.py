"""Register Xewton with the unified ``omni.physics.core`` interface.

Subclass of :class:`isaacsim.physics.newton.impl.register_simulation.NewtonSimulationRegistry`
that registers under engine name ``"Xewton"`` so the upstream ``"Newton"``
backend (loaded as a dependency) is not displaced.

The wrapped sim_fns / stage_update_fns from upstream operate purely against
the ``NewtonStage`` interface, so we reuse them as-is with our XewtonStage.
"""

from __future__ import annotations

import carb

from isaacsim.physics.newton.impl.register_simulation import NewtonSimulationRegistry
from isaacsim.physics.newton.impl.simulation_functions import NewtonSimulationFunctions
from isaacsim.physics.newton.impl.stage_update_functions import NewtonStageUpdateFunctions

from .xewton_stage import XewtonStage


class XewtonSimulationRegistry(NewtonSimulationRegistry):
    """Register Xewton as a separate physics engine alongside Newton."""

    ENGINE_NAME = "Xewton"

    def register_xewton(self, xewton_stage: XewtonStage) -> int | None:
        """Register Xewton as a physics simulation backend.

        Mirrors :meth:`NewtonSimulationRegistry.register_newton` but registers
        under :attr:`ENGINE_NAME` to coexist with the upstream ``Newton`` engine.
        """
        try:
            from omni.physics.core import Simulation, get_physics_interface, k_invalid_simulation_id

            self.newton_stage = xewton_stage

            self.sim_fns = NewtonSimulationFunctions(xewton_stage)
            self.stage_update_fns = NewtonStageUpdateFunctions(xewton_stage)

            xewton_stage.simulation_functions = self.sim_fns

            self.simulation = Simulation()

            sf = self.simulation.simulation_fns
            sf.initialize = self.sim_fns.initialize
            sf.close = self.sim_fns.close
            sf.get_attached_stage = self.sim_fns.get_attached_stage
            sf.simulate = self.sim_fns.simulate
            sf.fetch_results = self.sim_fns.fetch_results
            sf.check_results = self.sim_fns.check_results
            sf.flush_changes = self.sim_fns.flush_changes
            sf.pause_change_tracking = self.sim_fns.pause_change_tracking
            sf.is_change_tracking_paused = self.sim_fns.is_change_tracking_paused
            sf.subscribe_physics_contact_report_events = self.sim_fns.subscribe_physics_contact_report_events
            sf.unsubscribe_physics_contact_report_events = self.sim_fns.unsubscribe_physics_contact_report_events
            sf.get_simulation_time_steps_per_second = self.sim_fns.get_simulation_time_steps_per_second
            sf.get_simulation_timestamp = self.sim_fns.get_simulation_timestamp
            sf.get_simulation_step_count = self.sim_fns.get_simulation_step_count
            sf.subscribe_physics_on_step_events = self.sim_fns.subscribe_physics_on_step_events
            sf.unsubscribe_physics_on_step_events = self.sim_fns.unsubscribe_physics_on_step_events
            sf.is_capable_of_simulating = self.sim_fns.is_capable_of_simulating

            su = self.simulation.stage_update_fns
            su.start_simulation = self.stage_update_fns.start_simulation
            su.on_attach = self.stage_update_fns.on_attach
            su.on_detach = self.stage_update_fns.on_detach
            su.on_update = self.stage_update_fns.on_update
            su.on_resume = self.stage_update_fns.on_resume
            su.on_pause = self.stage_update_fns.on_pause
            su.on_reset = self.stage_update_fns.on_reset
            su.force_load_physics_from_usd = self.stage_update_fns.force_load_physics_from_usd
            su.release_physics_objects = self.stage_update_fns.release_physics_objects
            su.handle_raycast = self.stage_update_fns.handle_raycast
            su.reset_simulation = self.stage_update_fns.reset_simulation

            physics = get_physics_interface()
            self.simulation_id = physics.register_simulation(self.simulation, self.ENGINE_NAME)

            if self.simulation_id == k_invalid_simulation_id:
                carb.log_error(f"[{self.ENGINE_NAME}] Failed to register simulation with physics interface")
                return None

            self.sim_fns.simulation_id = self.simulation_id
            carb.log_info(f"[{self.ENGINE_NAME}] Registered with physics interface (ID: {self.simulation_id})")
            return self.simulation_id

        except Exception as e:
            carb.log_error(f"[{self.ENGINE_NAME}] Failed to register simulation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def unregister_xewton(self):
        """Unregister Xewton from the physics interface."""
        self.unregister_newton()
