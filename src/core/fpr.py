import numpy as np
import gdsfactory as gf

from taper import curved_taper, taper
from waveguide import waveguide
from ring import ring_arc
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec


def fpr(
    inputs: int = 1,
    outputs: int = 30,
    wi: float = 1.0,
    wo: float = 1.0,
    wa: float = 3.0,
    di: float = 3.0,
    do: float = 3.0,
    gi: float = 0.2,
    go: float = 0.2,
    lti: float = 20.0,
    lto1: float = 20.0,
    mi: float = 2.0,
    mo: float = 2.0,
    lwo: float = 10.0,
    lto2: float = 20.0,
    radius: float = 50.0,
    Ra: float = 100.0,
    center: tuple[float, float] = (0.0, 0.0),
    rotation_angle: float = 90.0,
    scale: float = 1.2,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Free propagation region (FPR).

    Args:
        inputs (int, optional): number of inputs.
        outputs (int, optional): number of outputs.
        wi (float, optional): width of the input waveguides.
        wo (float, optional): width of the output waveguides.
        wa (float, optional): width of the array waveguides.
        di (float, optional): spacing of the input tapers.
        do (float, optional): spacing of the output tapers.
        gi (float, optional): gap between input tapers.
        go (float, optional): gap between output tapers.
        lti (float, optional): length of the input tapers.
        lto1 (float, optional): length of the output tapers.
        mi (float, optional): m of input curved tapers.
        mo (float, optional): m of output curved tapers.
        lwo (float, optional): length of the single mode waveguides.
        lto2 (float, optional): length of the transition waveguides.
        radius (float, optional): radius of corrected bend waveguides.
        Ra (float, optional): radius of Rowland circle (length of FPR).
        center (tuple[float, float], optional): center of Rowland circle.
        rotation_angle (float, optional): rotation angle of FPR.
        scale (float, optional): scale index of boundary.
        cross_section (CrossSectionSpec, optional): cross_section function.
    """
    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None
    alpha = np.deg2rad(rotation_angle) # rotation theta
    
    FPR = gf.Component()

    pto = []
    w = do-go # temp w
    length_offset = 0.002
    theta_offset = 1e-4
    for i in range(1, outputs+1): # draw output tapers
        theta_term = -do*(i-(outputs+1)/2)/Ra
        theta_o =  theta_term + (np.pi/2-alpha)
        xo1 = Ra*np.sin(theta_o) + center[0]
        yo1 = Ra*np.cos(theta_o) + center[1]
        # xo2 = (Ra+lto1)*np.sin(theta_o) + center[0]
        # yo2 = (Ra+lto1)*np.cos(theta_o) + center[1]
        xo2_offset = (Ra+lto1-length_offset)*np.sin(theta_o) + center[0]
        yo2_offset = (Ra+lto1-length_offset)*np.cos(theta_o) + center[1]
        xo3 = (Ra+lto1+lwo)*np.sin(theta_o) + center[0]
        yo3 = (Ra+lto1+lwo)*np.cos(theta_o) + center[1]
        xo3_offset = (Ra+lto1+lwo+length_offset)*np.sin(theta_o) + center[0]
        yo3_offset = (Ra+lto1+lwo+length_offset)*np.cos(theta_o) + center[1]
        xo4 = (Ra+lto1+lwo+lto2)*np.sin(theta_o) + center[0]
        yo4 = (Ra+lto1+lwo+lto2)*np.cos(theta_o) + center[1]
        if theta_term > 0:
            xc1 = xo4 - radius*np.cos(theta_o)
            yc1 = yo4 + radius*np.sin(theta_o)
            L_offset = (-do*(1-(outputs+1)/2)/Ra-theta_term)*radius
            if alpha <= np.pi/2:
                xo5 = xc1 + radius*np.cos(alpha+theta_offset)
                yo5 = yc1 - radius*np.sin(alpha+theta_offset)
                xo6 = xo5 + L_offset*np.sin(alpha)
                yo6 = yo5 + L_offset*np.cos(alpha)
            else:
                xo5 = xc1 + radius*np.cos(alpha-np.pi/2+theta_offset)
                yo5 = yc1 + radius*np.sin(alpha-np.pi/2+theta_offset)
                xo6 = xo5 - L_offset*np.sin(alpha-np.pi/2)
                yo6 = yo5 + L_offset*np.cos(alpha-np.pi/2)
            ring1 = ring_arc(
                radius=radius,
                width=wa,
                angle_start=np.rad2deg(-theta_o),
                angle_stop=np.rad2deg(-alpha) if alpha <= np.pi/2 else np.rad2deg(alpha-np.pi/2),
                angle_resolution=0.5,
                center=(xc1, yc1),
                cross_section=cross_section
            )
        else:
            xc1 = xo4 + radius*np.cos(-theta_o)
            yc1 = yo4 + radius*np.sin(-theta_o)
            L_offset = (-do*(1-(outputs+1)/2)/Ra+theta_term)*radius
            if alpha <= np.pi/2:
                xo5 = xc1 - radius*np.cos(alpha-theta_offset)
                yo5 = yc1 + radius*np.sin(alpha-theta_offset)
                xo6 = xo5 + L_offset*np.sin(alpha)
                yo6 = yo5 + L_offset*np.cos(alpha)
            else:
                xo5 = xc1 - radius*np.cos(alpha-np.pi/2-theta_offset)
                yo5 = yc1 - radius*np.sin(alpha-np.pi/2-theta_offset)
                xo6 = xo5 - L_offset*np.sin(alpha-np.pi/2)
                yo6 = yo5 + L_offset*np.cos(alpha-np.pi/2)
            ring1 = ring_arc(
                radius=radius,
                width=wa,
                angle_start=np.rad2deg(np.pi-alpha) if alpha <= np.pi/2 else np.rad2deg(np.pi/2+alpha),
                angle_stop=np.rad2deg(np.pi-theta_o),
                angle_resolution=0.5,
                center=(xc1, yc1),
                cross_section=cross_section
            )
        
        taper_o, pto1, pto2 = curved_taper(
            m=mo,
            w1=w,
            w2=wo,
            length=lto1,
            rotation_angle=np.rad2deg(np.pi/2-theta_o),
            center=(xo1, yo1),
            cross_section=cross_section
        )
        pto.append(pto1)
        pto.append(pto2)

        wg1 = waveguide(
            start=(xo2_offset, yo2_offset), 
            end=(xo3_offset, yo3_offset),
            width=wo,
            cross_section=cross_section
        )

        taper1 = taper(
            w1=wo,
            w2=wa,
            length=lto2,
            rotate_angle=np.rad2deg(np.pi/2-theta_o),
            center=(xo3, yo3),
            cross_section=cross_section
        )

        wg_offset = waveguide(
            start=(xo5, yo5),
            end=(xo6, yo6),
            width=wa,
            cross_section=cross_section
        )
        
        FPR.add_ref(taper_o)
        FPR.add_ref(wg1)
        FPR.add_ref(taper1)
        FPR.add_ref(ring1)
        FPR.add_ref(wg_offset)

    pti = []
    w = di-gi # temp w
    for i in range(1, inputs+1):
        theta_i = di*(i-(inputs+1)/2)/(Ra/2) - (np.pi/2-alpha)
        xi = (Ra/2)*np.cos(alpha) + (Ra/2)*np.sin(theta_i) + center[0]
        yi = (Ra/2)*np.sin(alpha) - (Ra/2)*np.cos(theta_i) + center[1]
        taper_i, pti1, pti2 = curved_taper(
            m=mo,
            w1=w,
            w2=wi,
            length=lti,
            rotation_angle=np.rad2deg(-np.pi/2+theta_i),
            center=(xi, yi),
            cross_section=cross_section
        )
        pti.append(pti1)
        pti.append(pti2)
        FPR.add_ref(taper_i)
    
    # Boundary ref points
    theta_o1 = -do*(1-(outputs+1)/2)/Ra*scale + (np.pi/2-alpha)
    xbo1 = Ra*np.sin(theta_o1) + center[0]
    ybo1 = Ra*np.cos(theta_o1) + center[1]
    theta_o2 = -do*(outputs-(outputs+1)/2)/Ra*scale + (np.pi/2-alpha)
    xbo2 = Ra*np.sin(theta_o2) + center[0]
    ybo2 = Ra*np.cos(theta_o2) + center[1]
    theta_i1 = di*(1-(inputs+1)/2)/(Ra/2)*scale - (np.pi/2-alpha)
    xbi1 = (Ra/2)*np.cos(alpha) + (Ra/2)*np.sin(theta_i1) + center[0]
    ybi1 = (Ra/2)*np.sin(alpha) - (Ra/2)*np.cos(theta_i1) + center[1]
    theta_i2 = di*(inputs-(inputs+1)/2)/(Ra/2)*scale - (np.pi/2-alpha)
    xbi2 = (Ra/2)*np.cos(alpha) + (Ra/2)*np.sin(theta_i2) + center[0]
    ybi2 = (Ra/2)*np.sin(alpha) - (Ra/2)*np.cos(theta_i2) + center[1]

    boundary = []
    boundary.append([xbo1, ybo1])
    boundary.extend(pto)
    boundary.append([xbo2, ybo2])
    boundary.append([xbi1, ybi1])
    boundary.extend(pti)
    boundary.append([xbi2, ybi2])

    FPR.add_polygon(boundary, layer=layer)
    
    return FPR


if __name__ == "__main__":
    FPR = fpr(inputs=16, rotation_angle=60)
    FPR.show()
