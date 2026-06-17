import numpy as np
import gdsfactory as gf

from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


def curved_taper(
    m: float = 2.0,
    w1: float = 1.0,
    w2: float = 2.0,
    length: float = 10.0,
    n: int = 100,
    rotation_angle: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Curved taper waveguide.
    https://optics.ansys.com/hc/en-us/articles/360042799713-Curved-waveguide-taper-varFDTD-and-FDTD

    Args:
        m (float, optional): m of curved index.
        w1 (float, optional): short edge of the taper.
        w2 (float, optional): long edge of the taper.
        length (float, optional): length of the taper.
        n (int, optional): number of longitudinal discrete points.
        rotation_angle (float, optional): rotation angle of the taper.
        center (tuple[float, float], optional): center of short edge.
        cross_section (CrossSectionSpec, optional): cross_section function.
    """
    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None

    x = np.linspace(0, length, n)

    if w1 <= w2:
        alpha = (w1-w2) / (length**m)
        w = alpha*(length-x)**m + w2
    else:
        alpha = (w2-w1) / (length**m)
        w = alpha*x**m + w1

    top = np.column_stack((x, w/2))
    bot = np.column_stack((x, -w/2))

    vtx = np.vstack([top, bot[::-1]])

    # vtx = -vtx + np.array([length, 0])

    theta = np.deg2rad(rotation_angle)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    vtx = vtx @ rot_mat.T + np.array(center)

    c = gf.Component()
    c.add_polygon(vtx, layer=layer)
    return c, vtx[-1], vtx[0]


def taper(
    w1: float = 1.0,
    w2: float = 2.0,
    length: float = 10.0,
    rotate_angle: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Linear taper waveguide.
    https://optics.ansys.com/hc/en-us/articles/360042800413-Linear-waveguide-taper

    Args:
        w1 (float, optional): short edge of the taper.
        w2 (float, optional): long edge of the taper.
        length (float, optional): length of the taper.
        rotate_angle (float, optional): rotation angle of the taper.
        center (tuple[float, float], optional): center of short edge.
        cross_section (CrossSectionSpec, optional): cross_section function.
    """
    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None

    cx, cy = center
    theta = np.deg2rad(rotate_angle)
    s, c = np.sin(theta), np.cos(theta)

    v1 = [cx+w1/2*s, cy-w1/2*c]
    v2 = [cx-w1/2*s, cy+w1/2*c]
    v3 = [cx+length*c-w2/2*s, cy+length*s+w2/2*c]
    v4 = [cx+length*c+w2/2*s, cy+length*s-w2/2*c]
    
    c = gf.Component()
    c.add_polygon([v1, v2, v3, v4], layer=layer)
    return c


if __name__ == "__main__":
    taper = curved_taper(w1=3.0, w2=1.0, m=3, rotation_angle=60)
    # taper = taper(rotate_angle=60)
    taper.show()
