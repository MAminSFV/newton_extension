"""
Xewton test: rigid cube driven by a constant external body force in +X.

A prismatic joint locks the cube to the X axis (no Y/Z drift).
Gravity is disabled.  Each step, state.body_f is set to 5 N in +X.

Newton body_f spatial_vector layout: (fx, fy, fz, tx, ty, tz)
 — force first, torque second.  So 5 N in +X = (5, 0, 0, 0, 0, 0).

The XSolver doubles the linear-X component (index 0) of body_f, so:

Expected behavior
------------------
  Stock Newton : a =  5.00 m/s²  →  x(2s) ≈  10.00 m
  Xewton       : a = 10.00 m/s²  →  x(2s) ≈  20.00 m

How to run
----------
  1. Enable the minsim.physics.xewton extension in Isaac Sim.
  2. Open  Window and Script Editor.
  3. Paste or open this file, then click Run Script.
     Simulation auto-plays, runs ~2 s, prints results, and stops itself.
"""

import numpy as np
import omni.usd
import omni.timeline
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Sdf

# Stop the timeline up-front so re-running mid-test resets cleanly before we
# clear the stage (otherwise physics ticks against a half-built scene).
_timeline = omni.timeline.get_timeline_interface()
_timeline.stop()

# ── 1. Report active engine ─────────────────────────────────────────────────
try:
    import minsim.physics.xewton as xewton
    engine = xewton.get_active_physics_engine()
    ok = engine.lower() == "xewton"
    print(f"[xewton_test] engine = '{engine}'  {'✓ Xewton active' if ok else '✗ NOT Xewton — results will match Newton column'}")
except Exception as exc:
    print(f"[xewton_test] import failed: {exc}")
    xewton = None

# ── 0. Reset state and detach any callback registered by a previous run. ────
# subscribe_physics_step_events just does `physics_callbacks.append(callback)`
# and returns None — there is no token whose destructor cancels the sub. The
# _xewton_stage singleton (and its callback list) persists across script runs,
# so without an explicit unsubscribe each Run-Script appends another callback
# and Run N ends up with N callbacks racing on the same module globals.
try:
    _prev_cb = _registered_cb  # type: ignore[name-defined]
except NameError:
    _prev_cb = None

if _prev_cb is not None and xewton is not None:
    _prev_iface = xewton.acquire_physics_interface()
    if _prev_iface is not None:
        try:
            _prev_iface.unsubscribe_physics_step_events(_prev_cb)
        except ValueError:
            pass  # already gone (e.g. stage rebuilt by extension reload)

_registered_cb = None

_sim_t     = 0.0
_step      = 0
_body_idx  = None
_force_buf = None   # numpy array, built once when _body_idx is first resolved
_done      = False

# ── 2. Build stage ──────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().Clear()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")

# No gravity — force injected via body_f each step.
phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
phys_scene.CreateGravityMagnitudeAttr().Set(0.0)

# Static world anchor (no RigidBodyAPI = fixed/immovable).
# ArticulationRootAPI marks the root of the joint chain.
ANCHOR_PATH = "/World/Anchor"
anchor_prim = UsdGeom.Xform.Define(stage, ANCHOR_PATH).GetPrim()
UsdPhysics.ArticulationRootAPI.Apply(anchor_prim)

# Rigid cube: 0.5 m side, mass 1 kg, starts at rest at origin.
CUBE_PATH = "/World/Cube"
UsdGeom.Cube.Define(stage, CUBE_PATH).CreateSizeAttr().Set(0.5)
cube_prim = stage.GetPrimAtPath(CUBE_PATH)
UsdGeom.Xformable(cube_prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
UsdPhysics.RigidBodyAPI.Apply(cube_prim)
UsdPhysics.MassAPI.Apply(cube_prim).CreateMassAttr().Set(1.0)

# Prismatic joint: anchor (fixed) → cube (moving), X axis, unlimited travel.
JOINT_PATH = "/World/SlideJoint"
joint = UsdPhysics.PrismaticJoint.Define(stage, JOINT_PATH)
joint.GetBody0Rel().SetTargets([Sdf.Path(ANCHOR_PATH)])
joint.GetBody1Rel().SetTargets([Sdf.Path(CUBE_PATH)])
joint.CreateAxisAttr().Set("X")

# Ambient dome + a directional "sun" so the cube is visible during travel.
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(500.0)
sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun.CreateIntensityAttr().Set(3000.0)
sun.CreateAngleAttr().Set(1.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 30))

# Frame the viewport on the full ~20 m travel path: eye in -Y looking back at
# the centre of the run, elevated enough to see the cube against the ground plane.
try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(eye=[10.0, -25.0, 10.0], target=[10.0, 0.0, 0.0])

# ── 3. Step callback ────────────────────────────────────────────────────────
FORCE_N  = 5.0
MASS_KG  = 1.0
TARGET_T = 2.0      # seconds of simulation to run before stopping

A_NEWTON = FORCE_N / MASS_KG        #  5.00 m/s²
A_XEWTON = A_NEWTON * 2.0           # 10.00 m/s²

# Wrench values: (fx, fy, fz, tx, ty, tz) — force first, torque second.
_WRENCH_VALS = [FORCE_N, 0.0, 0.0, 0.0, 0.0, 0.0]  # 5 N in +X

def _on_step(dt: float) -> None:
    global _sim_t, _step, _body_idx, _force_buf, _done
    if _done:
        return

    newton_stage = xewton.acquire_stage() if xewton else None
    if newton_stage is None or newton_stage.state_0 is None or newton_stage.model is None:
        return

    # Resolve the cube's body index and build the force array in the same call,
    # so no step is wasted between resolution and first force injection.
    if _body_idx is None:
        try:
            _body_idx = list(newton_stage.model.body_label).index(CUBE_PATH)
        except ValueError:
            return
        # Reset timing so _sim_t measures from the first step where force is applied,
        # not from Play-press (Newton may take a few steps to initialize).
        _sim_t = 0.0
        _step  = 0
        # Build a full-length zeros array with the wrench at the correct body slot.
        # Assigning a single-element list would silently hit body 0, not _body_idx.
        _force_buf = np.zeros((newton_stage.model.body_count, 6), dtype=np.float32)
        _force_buf[_body_idx] = _WRENCH_VALS

    _sim_t += dt
    _step  += 1

    # Post-step callback: fires after simulate() returns, so this body_f is
    # consumed as state_in on the next step (one-step delay, negligible at 1 kHz).
    newton_stage.state_0.body_f.assign(_force_buf)

    if _sim_t < TARGET_T:
        return

    # ── TARGET_T reached: read final position, print, stop. ─────────────────
    _done = True

    # body_q is the pre-step position (one step behind _sim_t), a minor
    # sub-timestep discrepancy that is negligible over a 2 s run.
    body_q = newton_stage.state_0.body_q.numpy()
    x = float(body_q[_body_idx][0])

    x_newton = 0.5 * A_NEWTON * _sim_t ** 2
    x_xewton = 0.5 * A_XEWTON * _sim_t ** 2

    print()
    print("═" * 56)
    print(f"  steps = {_step}  |  t = {_sim_t:.4f} s  |  F = {FORCE_N} N  |  m = {MASS_KG} kg")
    print("─" * 56)
    print(f"  actual x          : {x:9.4f} m")
    print(f"  expected (Newton) : {x_newton:9.4f} m   (a = {A_NEWTON:.2f} m/s²)")
    print(f"  expected (Xewton) : {x_xewton:9.4f} m   (a = {A_XEWTON:.2f} m/s²)")
    if x_newton > 0:
        print(f"  ratio             : {x / x_newton:9.4f}x  (Xewton should be 2.0x)")
    print("═" * 56)

    omni.timeline.get_timeline_interface().stop()

if xewton:
    _iface = xewton.acquire_physics_interface()
    if _iface:
        _iface.subscribe_physics_step_events(_on_step)
        # Track the callback so the next Run-Script can unsubscribe it (see § 0).
        _registered_cb = _on_step
    else:
        print("[xewton_test] WARNING: could not acquire physics interface")
else:
    print("[xewton_test] WARNING: xewton not available")

print()
print(f"Auto-playing — will auto-stop after ~{TARGET_T:.0f} s and print results.")
print(f"  Stock Newton: a = {A_NEWTON:.2f} m/s²  →  x({TARGET_T:.0f}s) ≈ {0.5*A_NEWTON*TARGET_T**2:.2f} m")
print(f"  Xewton      : a = {A_XEWTON:.2f} m/s²  →  x({TARGET_T:.0f}s) ≈ {0.5*A_XEWTON*TARGET_T**2:.2f} m")

# Kick off the simulation immediately — no need to click Play.
_timeline.play()
