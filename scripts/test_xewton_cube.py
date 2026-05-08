"""
Xewton test: rigid cube driven by a constant external body force in +X.

A prismatic joint locks the cube to the X axis (no Y/Z drift).
Gravity is disabled.  Each step, state.body_f is set to a constant 5 N
in +X — the standard Newton API for external forces.

The XSolver doubles the linear-X component of body_f, so:

Expected behaviour
------------------
  Stock Newton : a =  5.00 m/s²  →  x ≈ 0.5 *  5.00 * t²
  Xewton       : a = 10.00 m/s²  →  x ≈ 0.5 * 10.00 * t²

How to run
----------
  1. Enable the minsim.physics.xewton extension in Isaac Sim.
  2. Open  Window → Script Editor.
  3. Paste or open this file, then click Run Script.
  4. Press ▶ Play. Watch the console for printed positions.
  5. Compare the printed x values against the two expected columns.
"""

import omni.usd
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

# No gravity — force is injected via body_f each step.
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
FORCE_N  = 5.0
A_NEWTON = FORCE_N          #  5.00 m/s²  (F / m = 5 / 1)
A_XEWTON = FORCE_N * 2.0    # 10.00 m/s²

# body_f spatial_vector layout: (wx, wy, wz, vx, vy, vz) — angular first.
# So constant 5 N in +X = (0, 0, 0, 5, 0, 0).
_wrench = wp.spatial_vector(0.0, 0.0, 0.0, FORCE_N, 0.0, 0.0)

_sim_t = 0.0
_step  = 0
_body_idx = None

def _on_step(dt: float) -> None:
    global _sim_t, _step, _body_idx
    _sim_t += dt
    _step  += 1

    newton_stage = xewton.acquire_stage() if xewton else None
    if newton_stage is None or newton_stage.state_0 is None or newton_stage.model is None:
        return

    # Resolve body index once.
    if _body_idx is None:
        try:
            _body_idx = list(newton_stage.model.body_label).index(CUBE_PATH)
        except ValueError:
            return

    # Inject constant force for the next step.
    # Newton reads state_in.body_f in solver.step(); setting it here
    # (post-step) populates it for the upcoming step after the state swap.
    newton_stage.state_0.body_f.assign([_wrench])

    if _step % 60 != 0:          # print once per ~second at 60 Hz
        return

    # body_q: wp.transform per body — [px, py, pz, qx, qy, qz, qw]
    body_q = newton_stage.state_0.body_q.numpy()
    x = float(body_q[_body_idx][0])

    x_newton = 0.5 * A_NEWTON * _sim_t ** 2
    x_xewton = 0.5 * A_XEWTON * _sim_t ** 2

    print(
        f"t={_sim_t:5.2f}s | "
        f"actual x={x:7.3f} m | "
        f"expected Newton={x_newton:7.3f}  Xewton={x_xewton:7.3f}"
    )

if xewton:
    _iface = xewton.acquire_physics_interface()
    if _iface:
        _sub = _iface.subscribe_physics_step_events(_on_step)
    else:
        print("[xewton_test] WARNING: could not acquire physics interface - callback not registered")
else:
    print("[xewton_test] WARNING: xewton not available - callback not registered")

print()
print("Stage ready — press Play.")
print(f"Cube constrained to X axis.  Constant force = {FORCE_N} N, mass = 1 kg.")
print(f"Stock Newton: a = {A_NEWTON:.2f} m/s²   Xewton: a = {A_XEWTON:.2f} m/s²")
