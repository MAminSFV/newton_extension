"""
Xewton test: rigid cube driven by a constant external body force in +X.

A prismatic joint locks the cube to the X axis (no Y/Z drift).
Gravity is disabled.  Each step, state.body_f is set to 5 N in +X.

Newton body_f spatial_vector layout: (fx, fy, fz, tx, ty, tz)
 — force first, torque second.  So 5 N in +X = (5, 0, 0, 0, 0, 0).

The XSolver doubles the linear-X component (index 0) of body_f, so:

Expected behaviour
------------------
  Stock Newton : a =  5.00 m/s²  →  x(2s) ≈  10.00 m
  Xewton       : a = 10.00 m/s²  →  x(2s) ≈  20.00 m

How to run
----------
  1. Enable the minsim.physics.xewton extension in Isaac Sim.
  2. Open  Window → Script Editor.
  3. Paste or open this file, then click Run Script.
  4. Press ▶ Play.  Simulation auto-stops at 2 s and prints results.
"""

import omni.usd
import omni.timeline
import warp as wp
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

# ── 1. Report active engine ────────────────────────────────────────────────
try:
    import minsim.physics.xewton as xewton
    engine = xewton.get_active_physics_engine()
    ok = engine.lower() == "xewton"
    print(f"[xewton_test] engine = '{engine}'  {'✓ Xewton active' if ok else '✗ NOT Xewton — results will match Newton column'}")
except Exception as exc:
    print(f"[xewton_test] import failed: {exc}")
    xewton = None

# ── 2. Build stage ─────────────────────────────────────────────────────────
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

# ── 3. Step callback ───────────────────────────────────────────────────────
FORCE_N   = 5.0
MASS_KG   = 1.0
SIM_LIMIT = 2.0          # stop and report after this many simulated seconds

A_NEWTON = FORCE_N / MASS_KG        #  5.00 m/s²
A_XEWTON = A_NEWTON * 2.0           # 10.00 m/s²

# body_f layout: (fx, fy, fz, tx, ty, tz) — force first.
# 5 N in +X = index 0.
_wrench = wp.spatial_vector(FORCE_N, 0.0, 0.0, 0.0, 0.0, 0.0)

_sim_t    = 0.0
_body_idx = None
_done     = False

def _on_step(dt: float) -> None:
    global _sim_t, _body_idx, _done
    if _done:
        return

    _sim_t += dt

    newton_stage = xewton.acquire_stage() if xewton else None
    if newton_stage is None or newton_stage.state_0 is None or newton_stage.model is None:
        return

    # Resolve body index once.
    if _body_idx is None:
        try:
            _body_idx = list(newton_stage.model.body_label).index(CUBE_PATH)
        except ValueError:
            return

    # Inject constant force for the next step (post-step callback fires after
    # the state swap, so state_0 is the upcoming state_in).
    newton_stage.state_0.body_f.assign([_wrench])

    if _sim_t < SIM_LIMIT:
        return

    # ── 2 s reached: read final position, print, stop. ──────────────────
    _done = True

    body_q = newton_stage.state_0.body_q.numpy()
    x = float(body_q[_body_idx][0])

    x_newton = 0.5 * A_NEWTON * SIM_LIMIT ** 2
    x_xewton = 0.5 * A_XEWTON * SIM_LIMIT ** 2

    print()
    print("═" * 52)
    print(f"  t = {_sim_t:.3f} s  |  F = {FORCE_N} N  |  m = {MASS_KG} kg")
    print("─" * 52)
    print(f"  actual x          : {x:8.4f} m")
    print(f"  expected (Newton) : {x_newton:8.4f} m   (a = {A_NEWTON:.2f} m/s²)")
    print(f"  expected (Xewton) : {x_xewton:8.4f} m   (a = {A_XEWTON:.2f} m/s²)")
    if x_newton > 0:
        print(f"  ratio             : {x / x_newton:8.4f}x  (Xewton should be 2.0x)")
    print("═" * 52)

    omni.timeline.get_timeline_interface().stop()

if xewton:
    _iface = xewton.acquire_physics_interface()
    if _iface:
        _sub = _iface.subscribe_physics_step_events(_on_step)
    else:
        print("[xewton_test] WARNING: could not acquire physics interface")
else:
    print("[xewton_test] WARNING: xewton not available")

print()
print("Stage ready — press Play.  Will auto-stop at 2 s and print results.")
print(f"  Stock Newton: a = {A_NEWTON:.2f} m/s²  →  x(2s) = {0.5*A_NEWTON*SIM_LIMIT**2:.2f} m")
print(f"  Xewton      : a = {A_XEWTON:.2f} m/s²  →  x(2s) = {0.5*A_XEWTON*SIM_LIMIT**2:.2f} m")
