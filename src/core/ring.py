import numpy as np
import gdsfactory as gf

from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


def ring_arc(
    radius: float = 10.0,
    width: float = 0.5,
    angle_start: float = 0.0,
    angle_stop: float = 90.0,
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
    if angle_stop < angle_start:
        raise ValueError(f"theta_stop={angle_stop} must be >= theta_start={angle_start}")

    
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    n = int(np.round((angle_stop - angle_start) / angle_resolution)) + 1
    t = np.linspace(angle_start, angle_stop, n) * np.pi / 180
    inner_points_x = center[0] + inner_radius * np.cos(t)
    inner_points_y = center[1] + inner_radius * np.sin(t)
    outer_points_x = center[0] + outer_radius * np.cos(t)
    outer_points_y = center[1] + outer_radius * np.sin(t)
    xpts = np.concatenate([inner_points_x, outer_points_x[::-1]])
    ypts = np.concatenate([inner_points_y, outer_points_y[::-1]])

    c = Component()
    c.add_polygon(points=list(zip(xpts, ypts, strict=False)), layer=layer)
    return c

