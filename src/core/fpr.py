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
    w0: float = 1.0,
    wi: float = 3.0,
    wa: float = 3.0,
    di: float = 4.0,
    do: float = 4.0,
    gap: float = 0.2,
    lti: float = 20.0,
    lto: float = 20.0,
    mi: float = 2.0,
    mo: float = 2.0,
    ls: float = 10.0,
    ltt: float = 20.0,
    radius: float = 50.0,
    Ra: float = 150.0,
    center: tuple[float, float] = (0.0, 0.0),
    rotation_angle: float = 90.0,
    scale: float = 1.2,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Free propagation region (FPR).

    Args:
        inputs (int, optional): number of inputs.
        outputs (int, optional): number of outputs.
        w0 (float, optional): width of single-mode waveguide.
        wi (float, optional): width of input aperture.
        wa (float, optional): width of the array waveguides.
        di (float, optional): spacing of the input tapers.
        do (float, optional): spacing of the output tapers.
        gap (float, optional): Gap width between waveguide apertures.
        lti (float, optional): length of the input tapers.
        lto (float, optional): length of the output tapers.
        mi (float, optional): m of input curved tapers.
        mo (float, optional): m of output curved tapers.
        ls (float, optional): length of the single mode waveguides.
        ltt (float, optional): length of the transition waveguides.
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
    
    c = gf.Component()

    pto = []
    ports = []
    length_offset = 0.002
    theta_offset = 1e-4
    half_pi = np.pi/2
    cx, cy = center
    for i in range(1, outputs+1): # draw output tapers
        theta_term = -do*(i-(outputs+1)/2)/Ra
        theta_o =  theta_term + (np.pi/2-alpha)
        st = np.sin(theta_o)
        ct = np.cos(theta_o)

        xo1 = Ra*st + cx
        yo1 = Ra*ct + cy
        # xo2 = (Ra+lto1)*st + cx
        # yo2 = (Ra+lto1)*ct + cy
        xo2_offset = (Ra+lto-length_offset)*st + cx
        yo2_offset = (Ra+lto-length_offset)*ct + cy
        xo3 = (Ra+lto+ls)*st + cx
        yo3 = (Ra+lto+ls)*ct + cy
        xo3_offset = (Ra+lto+ls+length_offset)*st + cx
        yo3_offset = (Ra+lto+ls+length_offset)*ct + cy
        xo4 = (Ra+lto+ls+ltt)*st + cx
        yo4 = (Ra+lto+ls+ltt)*ct + cy
        if theta_term > 0:
            xc1 = xo4 - radius*ct
            yc1 = yo4 + radius*st
            L_offset = (-do*(1-(outputs+1)/2)/Ra-theta_term)*radius
            if alpha <= half_pi:
                xo5 = xc1 + radius*np.cos(half_pi-alpha+theta_offset)
                yo5 = yc1 - radius*np.sin(half_pi-alpha+theta_offset)
                xo6 = xo5 + L_offset*np.sin(half_pi-alpha)
                yo6 = yo5 + L_offset*np.cos(half_pi-alpha)
            else:
                xo5 = xc1 + radius*np.cos(alpha-half_pi-theta_offset)
                yo5 = yc1 + radius*np.sin(alpha-half_pi-theta_offset)
                xo6 = xo5 - L_offset*np.sin(alpha-half_pi)
                yo6 = yo5 + L_offset*np.cos(alpha-half_pi)
            ring1 = ring_arc(
                radius=radius,
                width=wa,
                angle_start=np.rad2deg(-theta_o),
                angle_end=np.rad2deg(-(half_pi-alpha)) if alpha <= half_pi else np.rad2deg(alpha-half_pi),
                angle_resolution=0.5,
                center=(xc1, yc1),
                cross_section=cross_section
            )
        else:
            xc1 = xo4 + radius*ct
            yc1 = yo4 - radius*st
            L_offset = (-do*(1-(outputs+1)/2)/Ra+theta_term)*radius
            if alpha <= half_pi:
                xo5 = xc1 - radius*np.cos(half_pi-alpha-theta_offset)
                yo5 = yc1 + radius*np.sin(half_pi-alpha-theta_offset)
                xo6 = xo5 + L_offset*np.sin(half_pi-alpha)
                yo6 = yo5 + L_offset*np.cos(half_pi-alpha)
            else:
                xo5 = xc1 - radius*np.cos(alpha-half_pi+theta_offset)
                yo5 = yc1 - radius*np.sin(alpha-half_pi+theta_offset)
                xo6 = xo5 - L_offset*np.sin(alpha-half_pi)
                yo6 = yo5 + L_offset*np.cos(alpha-half_pi)
            ring1 = ring_arc(
                radius=radius,
                width=wa,
                angle_start=np.rad2deg(half_pi+alpha) if alpha <= half_pi else np.rad2deg(half_pi+alpha),
                angle_end=np.rad2deg(np.pi-theta_o),
                angle_resolution=0.5,
                center=(xc1, yc1),
                cross_section=cross_section
            )
        
        taper_o, pto1, pto2 = curved_taper(
            m=mo,
            w1=do-gap,
            w2=w0,
            length=lto,
            rotation_angle=np.rad2deg(half_pi-theta_o),
            center=(xo1, yo1),
            cross_section=cross_section
        )
        pto.append(pto1)
        pto.append(pto2)

        wg1 = waveguide(
            start=(xo2_offset, yo2_offset), 
            end=(xo3_offset, yo3_offset),
            width=w0,
            cross_section=cross_section
        )

        taper1 = taper(
            w1=w0,
            w2=wa,
            length=ltt,
            rotate_angle=np.rad2deg(half_pi-theta_o),
            center=(xo3, yo3),
            cross_section=cross_section
        )

        wg_offset = waveguide(
            start=(xo5, yo5),
            end=(xo6, yo6),
            width=wa,
            cross_section=cross_section
        )
        ports.append([xo6, yo6])

        c.add_ref(taper_o)
        c.add_ref(wg1)
        c.add_ref(taper1)
        c.add_ref(ring1)
        c.add_ref(wg_offset)

    pti = []
    for i in range(1, inputs+1):
        theta_i = di*(i-(inputs+1)/2)/(Ra/2) - (half_pi-alpha)
        xi = (Ra/2)*np.cos(alpha) + (Ra/2)*np.sin(theta_i) + cx
        yi = (Ra/2)*np.sin(alpha) - (Ra/2)*np.cos(theta_i) + cy
        taper_i, pti1, pti2 = curved_taper(
            m=mi,
            w1=wi,
            w2=w0,
            length=lti,
            rotation_angle=np.rad2deg(-half_pi+theta_i),
            center=(xi, yi),
            cross_section=cross_section
        )
        pti.append(pti1)
        pti.append(pti2)
        c.add_ref(taper_i)
    
    # Boundary ref points
    theta_o1 = -do*(1-(outputs+1)/2)/Ra*scale + (half_pi-alpha)
    xbo1 = Ra*np.sin(theta_o1) + cx
    ybo1 = Ra*np.cos(theta_o1) + cy
    theta_o2 = -do*(outputs-(outputs+1)/2)/Ra*scale + (half_pi-alpha)
    xbo2 = Ra*np.sin(theta_o2) + cx
    ybo2 = Ra*np.cos(theta_o2) + cy
    theta_i1 = di*(1-(inputs+1)/2)/(Ra/2)*scale - (half_pi-alpha)
    xbi1 = (Ra/2)*np.cos(alpha) + (Ra/2)*np.sin(theta_i1) + cx
    ybi1 = (Ra/2)*np.sin(alpha) - (Ra/2)*np.cos(theta_i1) + cy
    theta_i2 = di*(inputs-(inputs+1)/2)/(Ra/2)*scale - (half_pi-alpha)
    xbi2 = (Ra/2)*np.cos(alpha) + (Ra/2)*np.sin(theta_i2) + cx
    ybi2 = (Ra/2)*np.sin(alpha) - (Ra/2)*np.cos(theta_i2) + cy

    boundary = []
    boundary.append([xbo1, ybo1])
    boundary.extend(pto)
    boundary.append([xbo2, ybo2])
    boundary.append([xbi1, ybi1])
    boundary.extend(pti)
    boundary.append([xbi2, ybi2])

    c.add_polygon(boundary, layer=layer)
    
    return c, ports


if __name__ == "__main__":
    c = gf.Component()
    FPR, ports= fpr(
        inputs=16,
        outputs=32,
        rotation_angle=45,
        # Ra=300,
        # do=6,
        # di=6,
    )
    FPR_ref = c << FPR
    FPR_ref.mirror(p1=(400, 0), p2=(400, 100))
    FPR2, ports2= fpr(
        inputs=16,
        outputs=32,
        rotation_angle=45,
        # Ra=300,
        # do=6,
        # di=6,
    )
    c.add_ref(FPR2)
    c.show()
    print(ports[0][0])
