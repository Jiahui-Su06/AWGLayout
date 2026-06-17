import numpy as np
import gdsfactory as gf

from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


def waveguide(
    start: tuple[float, float] = (0.0, 0.0),
    end: tuple[float, float] = (10.0, 0.0),
    width: float = 0.5,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None

    dx, dy = end[0]-start[0], end[1]-start[1]
    length = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))

    section = gf.cross_section.strip(width=width, layer=layer)
    wg = gf.components.straight(
        length=length, cross_section=section
    )

    c = gf.Component()
    wg_ref = c << wg
    wg_ref.move(start)
    wg_ref.rotate(angle, center=start)
    return c


if __name__ == "__main__":
    wg = waveguide(
        start=(0, 0), 
        end=(4, 4),
        cross_section=gf.cross_section.strip(width=2.0)
    )
    wg.show()
