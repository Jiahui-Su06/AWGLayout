import gdsfactory as gf

from ring import ring_arc


c = gf.Component()
euler_bend = gf.components.bend_euler_all_angle(
    radius=50,
    width=1.0,
    angle=45,
    p=1,
    with_arc_floorplan=True,
    cross_section='strip',
    allow_min_radius_violation=False
)
euler_bend_ref = c.add_ref_off_grid(euler_bend)
euler_bend_ref.mirror(p1=(10, 0), p2=(10, 10))
# euler_bend_ref.rotate(30)
bend = ring_arc(
    radius=50,
    width=1.0,
    angle_start=-90,
    angle_end=-45,
    angle_resolution=0.5,
    center=(0, 50)
)
bend_ref = c.add_ref(bend)
bend_ref.mirror(p1=(10, 0), p2=(10, 10))

c.show()
