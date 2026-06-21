import numpy as np
import gdsfactory as gf

from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import CrossSectionSpec


def ring_arc(
    radius: float = 10.0,
    width: float = 0.5,
    angle_start: float = 0.0,
    angle_end: float = 90.0,
    center: tuple[float, float] = (0.0, 0.0),
    angle_resolution: float = 2.5,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Ring arc waveguide.

    Args:
        radius (float, optional): radius of the ring.
        width (float, optional): width of the ring.
        angle_start (float, optional): starting angle of the ring.
        angle_stop (float, optional): stopping angle of the ring.
        center (tuple[float, float], optional): center of the ring.
        angle_resolution (float, optional): angle resolution of the ring.
        cross_section (CrossSectionSpec, optional): cross_section function.
    """
    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None

    if radius <= 0:
        raise ValueError(f"radius={radius} must be > 0")
    if width <= 0:
        raise ValueError(f"width={width} must be > 0")
    if angle_end < angle_start:
        raise ValueError(f"theta_stop={angle_end} must be >= theta_start={angle_start}")

    
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    n = int(np.round((angle_end - angle_start) / angle_resolution)) + 1
    t = np.linspace(angle_start, angle_end, n) * np.pi / 180
    inner_points_x = center[0] + inner_radius * np.cos(t)
    inner_points_y = center[1] + inner_radius * np.sin(t)
    outer_points_x = center[0] + outer_radius * np.cos(t)
    outer_points_y = center[1] + outer_radius * np.sin(t)
    xpts = np.concatenate([inner_points_x, outer_points_x[::-1]])
    ypts = np.concatenate([inner_points_y, outer_points_y[::-1]])

    c = Component()
    c.add_polygon(points=list(zip(xpts, ypts, strict=False)), layer=layer)
    return c


def bend_euler(
    radius: float = 10.0,
    width: float = 1.0,
    angle_start: float = 0.0,
    angle_end: float = 45.0,
    center: tuple[float, float] = (0.0, 0.0),
    cross_section: CrossSectionSpec = "strip",
) -> ComponentAllAngle:
    if radius <= 0:
        raise ValueError(f"radius={radius} must be > 0")
    if width <= 0:
        raise ValueError(f"width={width} must be > 0")
    if angle_end < angle_start:
        raise ValueError(f"theta_stop={angle_end} must be >= theta_start={angle_start}")

    c = gf.Component()

    bend = gf.components.bends.bend_euler_all_angle(
        radius=radius,
        angle=angle_end-angle_start,
        p=1.0,
        width=width,
        cross_section=cross_section
    )
    bend_ref = c.add_ref_off_grid(bend)
    bend_ref.movey(-radius)
    bend_ref.rotate(90+angle_start)
    bend_ref.move(center)

    return c


if __name__ == "__main__":
    c = gf.Component()
    bend_euler = bend_euler(
        angle_start=30,
        angle_end=60,
        center=(10, 10)
    )
    c.add_ref_off_grid(bend_euler)
    c.show()
