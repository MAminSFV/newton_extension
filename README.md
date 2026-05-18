# XEWTON - Example Custom Newton Solver in Isaac Sim
How to extend/Augment Newton Sim Physics and integrate with the Issac Sim.

>[!Warning]
> This code was developed and tested with an experimental build of Isaac Sim. ([commit](https://github.com/isaac-sim/IsaacSim/commit/f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc) )


## How to Make A Custom Newton Solver

A Newton "solver" is anything that subclasses [`newton.solvers.SolverBase`](https://newton-physics.github.io/newton/stable/api/_generated/newton.solvers.SolverBase.html#newton.solvers.SolverBase) and implements one method — `step(state_in, state_out, control, contacts, dt)`. Newton's stepping loop calls it once per substep; everything else (collision detection, integration, contact resolution) is whatever you choose to do inside `step()`.

The demo solver in this repo — [`XSolver`](minsim.physics.xewton/minsim/physics/xewton/impl/xsolver.py) — illustrates the smallest non-trivial pattern: **pre-process the input state, delegate to an existing solver, restore the caller's state**. It doubles the linear-X component of every body's external wrench, hands the modified state off to `newton.solvers.SolverSemiImplicit`, then puts the caller's `body_f` array pointer back so the operation is idempotent.

### The contract

`SolverBase.step` is the only method you *must* implement. Its signature is fixed:

```python
def step(
    self,
    state_in:  newton.State,     # body_q, body_qd, body_f, joint_q, joint_qd, particle_*
    state_out: newton.State,     # write the new state here
    control:   newton.Control | None,   # joint_f, joint_target_pos, activations…
    contacts:  newton.Contacts | None,  # output of model.collide()
    dt: float,
) -> None: ...
```

The optional `notify_model_changed(flags: int)` hook lets the solver react to inertia/property edits at runtime — its default in `SolverBase` is a no-op, which is fine for most solvers.

The `newton.State` arrays a solver typically touches:

| Field | Type | Layout |
|---|---|---|
| `body_q`  | `wp.array[wp.transform]` | `(tx, ty, tz, qx, qy, qz, qw)` per body |
| `body_qd` | `wp.array[wp.spatial_vector]` | `(wx, wy, wz, vx, vy, vz)` — angular first, linear second |
| `body_f`  | `wp.array[wp.spatial_vector]` | `(fx, fy, fz, tx, ty, tz)` — **linear force first, torque second** |
| `joint_q`, `joint_qd` | `wp.array[float]` | generalised joint coords / velocities |

Note that `body_qd` and `body_f` use *different* component orders. XSolver doubles index 0 of `body_f` because that's `fx`.

### The XSolver pattern, walked through

```python
class XSolver(newton.solvers.SolverBase):
    def __init__(self, model, delegate=None):
        super().__init__(model)
        self.delegate = delegate or newton.solvers.SolverSemiImplicit(model)
        # 1. Allocate one persistent scratch buffer; reuse it every step.
        self._body_f_mod = wp.zeros(
            model.body_count, dtype=wp.spatial_vector, device=model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        # 2. Copy the caller's wrench into the scratch buffer.
        wp.copy(self._body_f_mod, state_in.body_f)

        # 3. Mutate the scratch buffer with a Warp kernel.
        wp.launch(_double_linear_x_kernel,
                  dim=self._body_f_mod.shape[0],
                  inputs=[self._body_f_mod],
                  device=self._body_f_mod.device)

        # 4. Temporarily swap the pointer, delegate, then restore.
        original_f = state_in.body_f
        state_in.body_f = self._body_f_mod
        try:
            self.delegate.step(state_in, state_out, control, contacts, dt)
        finally:
            state_in.body_f = original_f   # leave caller's state untouched
```

The Warp kernel itself is just the spatial-vector layout written out long-form:

```python
@wp.kernel
def _double_linear_x_kernel(body_f: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    f = body_f[tid]
    body_f[tid] = wp.spatial_vector(2.0 * f[0], f[1], f[2], f[3], f[4], f[5])
```

Three properties of this pattern worth borrowing:

- **Persistent scratch buffer.** Allocating inside `step()` would churn GPU memory at the physics rate. Allocate once in `__init__`, reuse forever.
- **Pointer swap, not mutation.** We never write back into `state_in.body_f` itself; we point `state_in.body_f` at our scratch buffer for the delegate call, then put it back. The caller's array is bit-identical before and after `step()`. Calling `step()` twice with the same input gives the same output — important if some other code path inadvertently re-runs the solver.
- **Delegation is optional but powerful.** XSolver wraps `SolverSemiImplicit`, but you could pre/post-process around any solver — Featherstone, MuJoCo, XPBD — to add force fields, force shaping, custom damping, etc., without re-implementing integration.

### Wiring a custom solver into Isaac Sim's Newton stage

A solver only runs if the stage knows to construct it. Isaac Sim's [`NewtonStage`](https://github.com/isaac-sim/IsaacSim/blob/develop/source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py) builds the solver via a `_get_solver(model, solver_cfg)` classmethod; subclass that to recognise your config:

- [`solver_config.py`](minsim.physics.xewton/minsim/physics/xewton/impl/solver_config.py) defines `XSolverConfig` with a single `solver_type: Literal["xsolver"]` discriminator.
- [`xewton_stage.py:57-62`](minsim.physics.xewton/minsim/physics/xewton/impl/xewton_stage.py#L57-L62) overrides `_get_solver`: if the config is `XSolverConfig`, build a `SolverSemiImplicit` delegate and wrap it in `XSolver`; otherwise fall through to the base implementation.
- [`xewton_config.py`](minsim.physics.xewton/minsim/physics/xewton/impl/xewton_config.py) makes `XSolverConfig` the default solver in the Xewton stage's config.

That's the entire integration on the Newton side — one config dataclass, one classmethod override.


### References
1. [Newton's FAQ](https://newton-physics.github.io/newton/stable/faq.html#is-newton-exposed-and-accessible-in-isaac-lab-and-isaac-sim) on Multi-physics and Custom solvers
2. [Newton's Docs on Solvers](https://newton-physics.github.io/newton/stable/api/newton_solvers.html#newton-solvers) and [BaseSolver class](https://newton-physics.github.io/newton/stable/api/_generated/newton.solvers.SolverBase.html#newton.solvers.SolverBase)


## How to Integrate with Isaac Sim Using Newton's Omniverse Kit Extensions

### Extension Installation

The extension lives in [`minsim.physics.xewton/`](minsim.physics.xewton/). It is a self-contained Omniverse Kit extension — no build step is required, just point Isaac Sim's extension manager at the folder.

1. **Clone the repo** somewhere stable (the path will be referenced by Isaac Sim):
   ```bash
   git clone <repo-url> ~/code/newton_extension
   ```
2. **Launch Isaac Sim** (6.0 early-access or later, see warning above).
3. **Register the extension search path.** In Isaac Sim, open `Window → Extensions`, click the ⚙ (gear) icon, and add the **absolute path to the parent folder containing the extension** (i.e. the repo root, not `minsim.physics.xewton/` itself) to "Extension Search Paths". For the clone above:
   ```
   /home/<you>/code/newton_extension
   ```
4. **Enable the extension.** Switch to the "Third Party" tab, search for `minsim.physics.xewton` (title: *Xewton: Amin's Custom Newton Physics*), toggle it on, and tick "Autoload" if you want it active on every Isaac Sim launch.
5. **Verify.** Open Window → Script Editor and run:
   ```python
   import minsim.physics.xewton as xewton
   for name, active in xewton.get_available_physics_engines(verbose=True):
       print(name, active)
   ```
   You should see `PhysX`, `Newton`, and `Xewton` listed, with `Xewton` active (the extension auto-switches on startup — see `auto_switch_on_startup = true` in [`extension.toml`](minsim.physics.xewton/config/extension.toml#L34)).

If you prefer a non-interactive launch, pass these to the Isaac Sim binary directly:
```
--ext-folder /home/<you>/code/newton_extension
--enable minsim.physics.xewton
```

### Solver Demo

[`scripts/test_xewton_cube.py`](scripts/test_xewton_cube.py) is a self-contained behavioural test of the doubled-force solver:

- Builds a 1 kg cube on a prismatic X-joint anchored at the origin, gravity off, no collisions in the way.
- Each physics step, injects a constant **5 N in +X** via `state_0.body_f`.
- Auto-plays for ~2 s, then prints the cube's final X position.

Stock Newton would predict `x(2s) = ½·5·2² = 10.00 m`. The XSolver doubles +X forces, so the cube actually accelerates at 10 m/s² and lands at **~20.01 m** — visibly different and easy to confirm. The script prints a ratio line; `≈ 2.0×` means the custom solver is active.

To run it: enable the extension as above, then in Window → Script Editor open `scripts/test_xewton_cube.py` and click *Run Script*. The script subscribes a physics-step callback, calls `omni.timeline.play()` itself, stops the timeline after `TARGET_T = 2.0 s` of simulated time, and prints results. Re-running it in the same session unsubscribes the previous callback first, so the test is repeatable without restarting Isaac Sim.

### How Integration was Done

The pattern is "subclass the upstream Newton extension classes, override the smallest possible surface, register under a new engine name." Six classes are involved, all in [`minsim.physics.xewton/minsim/physics/xewton/impl/`](minsim.physics.xewton/minsim/physics/xewton/impl/):

| Class | File | Parent | Purpose |
|---|---|---|---|
| [`XSolver`](minsim.physics.xewton/minsim/physics/xewton/impl/xsolver.py#L64) | `xsolver.py` | `newton.solvers.SolverBase` | The custom solver itself; doubles +X linear forces, delegates to `SolverSemiImplicit`. |
| [`XSolverConfig`](minsim.physics.xewton/minsim/physics/xewton/impl/solver_config.py#L18) | `solver_config.py` | `isaacsim.physics.newton.NewtonSolverConfig` | Discriminator dataclass — `solver_type = "xsolver"`. |
| [`XewtonConfig`](minsim.physics.xewton/minsim/physics/xewton/impl/xewton_config.py#L18) | `xewton_config.py` | `isaacsim.physics.newton.NewtonConfig` | Top-level config; defaults `solver_cfg` to `XSolverConfig`. |
| [`XewtonStage`](minsim.physics.xewton/minsim/physics/xewton/impl/xewton_stage.py#L20) | `xewton_stage.py` | `isaacsim.physics.newton.impl.newton_stage.NewtonStage` | Overrides `_get_solver` to build `XSolver` when the config calls for it; also wraps `model.collide()` for older Newton API compatibility. |
| [`XewtonSimulationRegistry`](minsim.physics.xewton/minsim/physics/xewton/impl/xewton_register.py#L22) | `xewton_register.py` | `isaacsim.physics.newton.impl.register_simulation.NewtonSimulationRegistry` | Registers under the unified `omni.physics.core` interface as engine name `"Xewton"`, alongside `"Newton"` and `"PhysX"`. |
| [`XewtonSimExtension`](minsim.physics.xewton/minsim/physics/xewton/impl/extension.py#L75) | `extension.py` | `omni.ext.IExt` | Kit lifecycle entry point — on startup builds `XewtonConfig`, instantiates `XewtonStage`, registers it, and optionally switches the active engine. |

The flow at startup is: `XewtonSimExtension.on_startup` → `XewtonConfig()` → `XewtonStage(cfg=...)` → `XewtonSimulationRegistry().register_xewton(...)` → when the user hits Play, `XewtonStage.initialize_newton()` calls `XewtonStage._get_solver(...)` which returns `XSolver(model, delegate=SolverSemiImplicit(model))` — and from then on each physics tick calls `XSolver.step(...)`.

The Kit-side surface is just the [`config/extension.toml`](minsim.physics.xewton/config/extension.toml): name, deps (`isaacsim.physics.newton`, `omni.physics`, `usdrt.scenegraph`, `omni.warp.core`, `omni.isaac.ml_archive`), and a single `[[python.module]]` block with `path = "."` so Kit finds `minsim/physics/xewton/__init__.py` at the repo-relative root rather than in the default `config/` directory.

### Caveats
1. Needless to say, Isaac 6.0 is early access hence unstable, incomplete, and can have bugs.
2. Only a limited set of Newton solvers are integrated. (namely hydro-elastic contact is not integrated)
3. Not all of the Newton integration code is focused in the three Kit extensions related to Newton. The approach taken in this repo is limited to depending on those extensions. However, some user-facing features or warnings might need changes to other experimental kit extensions from Isaac Sim for deeper support. (see Simulation Manager and general schema handling)
   - **Example: `SimulationManager` does not recognise third-party engines.** When Xewton is active you'll see `[Warning] Unknown engine 'xewton', defaulting to PhysX` in the console. The cause is [`simulation_manager.py:405-414`](https://github.com/isaac-sim/IsaacSim/blob/develop/source/extensions/isaacsim.core.simulation_manager/python/impl/simulation_manager.py#L405-L414), whose `_create_physics_scene` hardcodes `if engine == "physx" / elif engine == "newton"` with no extension hook; anything else falls into the `else:` branch and gets wrapped in a `PhysxScene`. The actual physics still runs through the unified `omni.physics.core` interface (XSolver's 2× result in the demo proves this), so scripts that talk directly to `xewton.acquire_stage()` / `xewton.acquire_physics_interface()` are unaffected. Higher-level wrappers that consume the `SimulationManager`'s scene object (e.g. fabric-backed transforms, `isaacsim.core.api.World`) may follow PhysX-flavoured code paths — fixing that needs an upstream change to add an engine-registration hook.


### References
1. [Isaac's Newton Kit extension implementation](https://github.com/isaac-sim/IsaacSim/tree/develop/source/extensions/isaacsim.physics.newton)
2. [Omniverse Kit Docs](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/kit_overview.html)


### Personal Take-aways
- Omniverse Kit Developer tools and experience is non-standard and it is hard to work with. All things considered the documentation and examples repo is quite nice and illuminating and help with the starting point, however, having a build and test environment only for the extensions is not easy. The easiest way for me to test my code was doing an end to end test using an Isaac Sim instance and loading the extension to see if the extension can be loaded properly or not. Hopefully NVidia will improve on this in the future.
- AI Coding Agents have a very difficult time to figure out Omniverse and Isaac Sim. Newton on the other hand, is very well structured and AI friendly. Omniverse's history and defacto stack choices not only slows the Human developers but also confuses AI agents a lot.
- Isaac Sim is just slow to start up and the user experience is a bit clunky.
- It seems that Isaac Lab has [a different way of integrating Newton](https://github.com/isaac-sim/IsaacLab/tree/develop/source/isaaclab_newton/isaaclab_newton/physics) into the application. It looks like Isaac Lab is not following Omniverse's best practices and the team is hacking around it and taking a more opinionated (Pythonic?) approach.
