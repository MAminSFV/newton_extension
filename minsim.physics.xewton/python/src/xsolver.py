"""
XSolver: a demonstration custom Newton solver.

Semantics
---------
For every rigid body in the model, before each physics step the linear-X
component of the external wrench `state_in.body_f[i]` is effectively
replaced by `2 * linear_X`. All other components are untouched. The
modified wrench is then passed to a delegate solver, which performs the
actual integration.

The modification is done on a SCRATCH copy -- the caller's `state_in.body_f`
is not mutated. This keeps the solver idempotent: calling `step()` twice
with the same input produces consistent output, and repeated applications
across a simulation loop don't compound the doubling.

Conventions
-----------
- `newton.State.body_f`  : wp.array of wp.spatial_vector, length = model.body_count.
- `newton.State.body_q`  : wp.array of wp.transform (pos.xyz, quat.xyzw = 7 floats).
- `newton.State.body_qd` : wp.array of wp.spatial_vector (spatial twist).
- `wp.spatial_vector`    : layout (vx, vy, vz, wx, wy, wz),
                            i.e. linear 3-vector first, angular 3-vector second.
                            Index 0 is linear X.

References
----------
- Newton solvers:      https://newton-physics.github.io/newton/api/newton_solvers.html
- Newton conventions:  https://newton-physics.github.io/newton/concepts/conventions.html
"""

from __future__ import annotations

from typing import Optional

import warp as wp
import newton


# --------------------------------------------------------------------------- #
# Warp kernel
# --------------------------------------------------------------------------- #


@wp.kernel
def _double_linear_x_kernel(body_f: wp.array(dtype=wp.spatial_vector)):
    """In-place: multiply the linear-X component of each wrench by 2.

    spatial_vector layout is (vx, vy, vz, wx, wy, wz); index 0 = vx.
    """
    tid = wp.tid()
    f = body_f[tid]
    body_f[tid] = wp.spatial_vector(
        2.0 * f[0], f[1], f[2],  # linear: +X doubled, Y/Z unchanged
        f[3], f[4], f[5],        # angular unchanged
    )


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


class XSolver(newton.solvers.SolverBase):
    """
    Custom solver that doubles rigid-body +X linear forces, then delegates.

    Parameters
    ----------
    model : newton.Model
        The finalized Newton model. Must match the model passed to `step()`
        and to the delegate solver.
    delegate : newton.solvers.SolverBase, optional
        The solver that performs the actual integration. Defaults to
        ``newton.solvers.SolverSemiImplicit(model)``, the cheapest reference
        integrator.

    Notes
    -----
    - The original ``state_in.body_f`` is not mutated. A persistent scratch
      buffer is allocated at construction and reused each step.
    - Contacts must be populated by the caller via ``model.collide(state_in,
      contacts)`` prior to calling ``step()``, same as any other Newton solver.
    """

    def __init__(
        self,
        model: newton.Model,
        delegate: Optional[newton.solvers.SolverBase] = None,
    ):
        super().__init__(model)

        self.delegate: newton.solvers.SolverBase = (
            delegate if delegate is not None else newton.solvers.SolverSemiImplicit(model)
        )

        # Persistent scratch buffer for the modified wrench.
        # Allocated once; reused every step.
        self._body_f_mod: Optional[wp.array] = None
        if model.body_count > 0:
            self._body_f_mod = wp.zeros(
                model.body_count,
                dtype=wp.spatial_vector,
                device=model.device,
            )

    # ------------------------------------------------------------------ #
    # SolverBase contract
    # ------------------------------------------------------------------ #

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ) -> None:
        """Advance the simulation by `dt`, doubling +X forces first.

        Reads:   state_in.body_f (not mutated)
        Writes:  state_out (via the delegate)
        """
        have_bodies = (
            self._body_f_mod is not None
            and state_in.body_f is not None
            and state_in.body_f.shape[0] > 0
        )

        if not have_bodies:
            # No rigid bodies with wrench data -- nothing to double. Pass through.
            self.delegate.step(state_in, state_out, control, contacts, dt)
            return

        # 1. Copy caller's body_f into our scratch buffer.
        wp.copy(self._body_f_mod, state_in.body_f)

        # 2. Double the linear-X component in place (on the scratch buffer only).
        wp.launch(
            kernel=_double_linear_x_kernel,
            dim=self._body_f_mod.shape[0],
            inputs=[self._body_f_mod],
            device=self._body_f_mod.device,
        )

        # 3. Temporarily swap the scratch buffer into state_in; delegate; restore.
        original_f = state_in.body_f
        state_in.body_f = self._body_f_mod
        try:
            self.delegate.step(state_in, state_out, control, contacts, dt)
        finally:
            state_in.body_f = original_f
