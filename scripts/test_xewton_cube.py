"""
Xewton test: rigid cube accelerating in +X under simulated gravity.

The XSolver doubles the linear-X component of every body wrench each step,
so a force applied in +X is effectively doubled compared to stock Newton.

Expected behaviour
------------------
  Stock Newton : a =  9.81 m/s²  →  x ≈ 0.5 * 9.81  * t²
  Xewton       : a = 19.62 m/s²  →  x ≈ 0.5 * 19.62 * t²

How to run
----------
  1. Enable the minsim.physics.xewton extension in Isaac Sim.
  2. Open  Window → Script Editor.
  3. Paste or open this file, then click Run Script.
  4. Press ▶ Play. Watch the console for printed positions.
  5. Compare the printed x values against the two expected columns.
"""

import carb
import omni.usd
import omni.physx
from pxr import UsdGeom, UsdPhysics, Gf

# ── 1. Report active engine ────────────────────────────────────────────────
try:
    import minsim.physics.xewton as xewton
    engine = xewton.get_active_physics_engine()
    ok = engine.lower() == "xewton"
    print(f"[xewton_test] engine = '{engine}'  {'✓ Xewton active' if ok else '✗ NOT Xewton — results will match Newton column'}")
except Exception as exc:
    print(f"[xewton_test] import failed: {exc}")

# ── 2. Build stage ─────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().Clear()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")

# "Gravity" points in +X at 9.81 m/s².
# XSolver doubles the +X wrench component → effective acceleration = 19.62 m/s².
# Y and Z forces are zero, so the cube moves in X only.
phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(1, 0, 0))
phys_scene.CreateGravityMagnitudeAttr().Set(9.81)

# Rigid cube, 0.5 m side, mass 1 kg, starts at rest at origin.
CUBE_PATH = "/World/Cube"
UsdGeom.Cube.Define(stage, CUBE_PATH).CreateSizeAttr().Set(0.5)
prim = stage.GetPrimAtPath(CUBE_PATH)
UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
UsdPhysics.CollisionAPI.Apply(prim)
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(1.0)

# ── 3. Step callback ───────────────────────────────────────────────────────
_sim_t = 0.0
_step  = 0

def _on_step(dt: float) -> None:
    global _sim_t, _step
    _sim_t += dt
    _step  += 1
    if _step % 60 != 0:          # print once per ~second at 60 Hz
        return

    pos = prim.GetAttribute("xformOp:translate").Get()
    if pos is None:
        return

    x = pos[0]
    x_newton = 0.5 * 9.81  * _sim_t ** 2
    x_xewton = 0.5 * 19.62 * _sim_t ** 2

    print(
        f"t={_sim_t:5.2f}s | "
        f"actual x={x:7.3f} m | "
        f"expected Newton={x_newton:7.3f}  Xewton={x_xewton:7.3f}"
    )

_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(_on_step)

print()
print("Stage ready — press ▶ Play.")
print("The cube should accelerate in +X only (Y=0, Z=0).")
print("With Xewton: matches the Xewton column (~19.62 m/s²).")
print("With stock Newton: matches the Newton column (~9.81 m/s²).")
