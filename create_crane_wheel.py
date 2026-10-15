import cadquery as cq

# Base dimensions (in mm)
OUTER_DIAMETER = 300.0       # Tread diameter (Диаметр поверхности катания)
FLANGE_DIAMETER = 340.0      # Flange diameter (Диаметр реборд)
TOTAL_WIDTH = 90.0           # Overall width of rim / wheel
FLANGE_WIDTH = 15.0          # Thickness of each flange
TREAD_WIDTH = TOTAL_WIDTH - 2 * FLANGE_WIDTH  # 60 mm
BORE_DIAMETER = 50.0         # Shaft hole diameter (Диаметр отверстия под вал)
HUB_DIAMETER = 110.0         # Hub outer diameter
WEB_THICKNESS = 30.0         # Web (диск) thickness

# GOST 23360-78 for d = 50 mm
KEYWAY_WIDTH = 14.0          # b
KEYWAY_DEPTH_HUB = 3.8       # t2 (depth in hub above bore wall)

def build_crane_wheel():
    # Construct 2D profile for revolution around X axis or Z axis
    # Let's create by revolving a profile or building 3D primitives and boolean operations.

    # 1. Main cylindrical body of wheel (outer rim and hub profile)
    # Revolve profile in XY plane around X axis:
    # Radius profile from hub to flange

    # Alternatively, create cylinder for wheel body, cut/extrude features:
    # Outer cylinder (flanges): R = 170 (D = 340), L = 90
    wheel = cq.Workplane("XY").cylinder(TOTAL_WIDTH, FLANGE_DIAMETER / 2.0)

    # Cut tread groove: Tread radius = 150 (D = 300), width = 60
    tread_cut = cq.Workplane("XY").cylinder(TREAD_WIDTH, FLANGE_DIAMETER / 2.0)
    tread_solid = cq.Workplane("XY").cylinder(TREAD_WIDTH, OUTER_DIAMETER / 2.0)
    tread_ring = tread_cut.cut(tread_solid)

    wheel = wheel.cut(tread_ring)

    # Cut side recesses for web (диск колеса между ступицей и ободом)
    # Web thickness = 30, so recess depth on each side = (90 - 30) / 2 = 30 mm
    # Inner rim radius = 125 (D = 250), outer hub radius = 55 (D = 110)
    RIM_INNER_DIAMETER = 240.0
    recess_outer_r = RIM_INNER_DIAMETER / 2.0
    recess_inner_r = HUB_DIAMETER / 2.0
    recess_depth = (TOTAL_WIDTH - WEB_THICKNESS) / 2.0 # 30 mm

    recess_cylinder_outer = cq.Workplane("XY").workplane(offset=WEB_THICKNESS/2.0).cylinder(recess_depth, recess_outer_r, centered=(True, True, False))
    recess_cylinder_inner = cq.Workplane("XY").workplane(offset=WEB_THICKNESS/2.0).cylinder(recess_depth, recess_inner_r, centered=(True, True, False))
    recess_ring_front = recess_cylinder_outer.cut(recess_cylinder_inner)

    recess_cylinder_outer_back = cq.Workplane("XY").workplane(offset=-WEB_THICKNESS/2.0 - recess_depth).cylinder(recess_depth, recess_outer_r, centered=(True, True, False))
    recess_cylinder_inner_back = cq.Workplane("XY").workplane(offset=-WEB_THICKNESS/2.0 - recess_depth).cylinder(recess_depth, recess_inner_r, centered=(True, True, False))
    recess_ring_back = recess_cylinder_outer_back.cut(recess_cylinder_inner_back)

    wheel = wheel.cut(recess_ring_front).cut(recess_ring_back)

    # Central bore hole (50 mm)
    bore = cq.Workplane("XY").cylinder(TOTAL_WIDTH + 10.0, BORE_DIAMETER / 2.0)
    wheel = wheel.cut(bore)

    # Keyway GOST 23360-78: width = 14 mm, depth t2 = 3.8 mm into hub
    # Keyway top edge is at R_bore + t2 = 25 + 3.8 = 28.8 mm from center
    keyway_box = (
        cq.Workplane("XY")
        .workplane(offset=-TOTAL_WIDTH)
        .rect(KEYWAY_WIDTH, (BORE_DIAMETER/2.0 + KEYWAY_DEPTH_HUB) * 2)
        .extrude(TOTAL_WIDTH * 2)
    )
    # Position box so top is at Y = 28.8 mm
    # Box centered at (0, 28.8 - (25+3.8))
    # Actually, if rect is 14 x H, center at Y = 28.8 - H/2:
    H_box = 20.0
    keyway = (
        cq.Workplane("XY")
        .center(0, BORE_DIAMETER / 2.0 + KEYWAY_DEPTH_HUB - H_box / 2.0)
        .rect(KEYWAY_WIDTH, H_box)
        .extrude(TOTAL_WIDTH + 10.0, both=True)
    )

    wheel = wheel.cut(keyway)

    return wheel

if __name__ == "__main__":
    wheel = build_crane_wheel()
    cq.exporters.export(wheel, "crane_wheel.step")
    cq.exporters.export(wheel, "crane_wheel.stl")
    print("3D Model exported successfully to crane_wheel.step and crane_wheel.stl")
