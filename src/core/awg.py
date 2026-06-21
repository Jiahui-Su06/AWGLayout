import numpy as np
import gdsfactory as gf

from gdsfactory.typings import CrossSectionSpec
from gdsfactory.component import Component

from fpr import fpr
from ring import bend_euler
from waveguide import waveguide


def awg(
    # Optical parameters
    m: float = 40,
    wl: float = 1.55012,
    neff: float = 1.6587,
    # Core parameters
    w0: float = 1.2,
    wa: float = 3.0,
    Ni: float = 8,
    No: float = 8,
    Na: float = 22,
    Ra: float = 124.428,
    di: float = 2.5,
    do: float = 2.5,
    da: float = 4.5,
    # Aperture parameters
    wi: float = 2.0,
    wo: float = 2.0,
    gap: float = 0.180,
    # Geometry parameters
    rotation_angle: float = 100,
    L1_0: float = 150,
    L2_0: float = 0,
    L3_0: float = 0,
    deltaW: float = 7,
    # Taper transition parameters
    lti: float = 20,
    lto: float = 20,
    lta: float = 30,
    mi: float = 3.0,
    mo: float = 3.0,
    ma: float = 3.0,
    ls: float = 20,
    ltt: float = 20,
    # Bend parameters
    Rm: float = 220,
    Re: float = 60,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Array Waveguide Grating. (AWG)

    Args:
        m (float, optional): Diffraction order.
        wl (float, optional): Operating central wavelength. (μm)
        neff (float, optional): Effective refractive index of arrayed waveguides.
        w0 (float, optional): Width of single-mode waveguide.
        wa (float, optional): Width of arrayed waveguide.
        Ni (float, optional): Number of input channels.
        No (float, optional): Number of output channels.
        Na (float, optional): Number of arrayed waveguides.
        Ra (float, optional): Radius of Rowland circle.
        di (float, optional): Spacing of the input aperture.
        do (float, optional): Spacing of the output aperture.
        da (float, optional): Spacing of arrayed aperture.
        wi (float, optional): Width of input aperture.
        wo (float, optional): Width of output aperture.
        gap (float, optional): Gap width between waveguide apertures.
        rotation_angle (float, optional): Rotation angle of input FPR.
        L1_0 (float, optional): Initial length of first waveguide.
        L2_0 (float, optional): Initial length of second waveguide.
        L3_0 (float, optional): Initial length of third waveguide.
        deltaW (float, optional): Horizontal spacing between arrayed waveguides.
        lti (float, optional): Length of input taper.
        lto (float, optional): Length of output taper.
        lta (float, optional): Length of arrayed taper.
        mi (float, optional): Curved factor of input curved taper.
        mo (float, optional): Curved factor of output curved taper.
        ma (float, optional): Curved factor of arrayed curved taper.
        ls (float, optional): Length of single-mode waveguide.
        ltt (float, optional): Length of transmission taper.
        Rm (float, optional): Radius of multi-mode bend wavguide.
        Re (float, optional): Radius of Euler bend waveguide.
        cross_section (CrossSectionSpec, optional): Cross_section function.
    """
    xs = gf.get_cross_section(cross_section)
    layer = xs.layer
    assert layer is not None

    rotation_theta = np.deg2rad(rotation_angle)

    c = gf.Component()

    # Input FPR (left)
    FPR_i, ports_i = fpr(
        inputs=Ni,
        outputs=Na,
        w0=w0,
        wi=wi,
        wa=wa,
        di=di,
        do=da,
        gap=gap,
        lti=lti,
        lto=lta,
        mi=mi,
        mo=ma,
        ls=ls,
        ltt=ltt,
        radius=Rm,
        Ra=Ra,
        center=(0, 0),
        rotation_angle=rotation_angle,
        cross_section=cross_section,
        scale=1.5
    )
    c.add_ref(FPR_i)

    x_center = 0.0
    half_pi = np.pi/2

    for i in range(1, Na+1):
        if i == 1:
            x1 = ports_i[0][0] + L1_0*np.cos(rotation_theta)
            y1 = ports_i[0][1] + L1_0*np.sin(rotation_theta)
            if rotation_theta <= half_pi:
                xc1 = x1 - Re*np.sin(rotation_theta)
                yc1 = y1 + Re*np.cos(rotation_theta)
                x2 = xc1 + Re
                y2 = yc1
                angle_start = rotation_angle-90
                angle_end = 0
            else:
                xc1 = x1 + Re*np.sin(rotation_theta)
                yc1 = y1 - Re*np.cos(rotation_theta)
                x2 = xc1 - Re
                y2 = yc1
                angle_start = 180
                angle_end = 90+rotation_angle
            x3 = x2
            y3 = y2 + L2_0
            xc2 = x3 + Re
            yc2 = y3
            x4 = xc2
            y4 = yc2 + Re
            x5 = x4 + L3_0
            y5 = y4
            x_center = x5
        else:
            if rotation_theta <= half_pi:
                L1 = (
                    x_center-L3_0-Re-deltaW*(i-1)
                    -ports_i[i-1][0]
                    -Re*(1-np.sin(rotation_theta))
                ) / np.cos(rotation_theta)
                L3 = (
                    L3_0
                    +deltaW*(i-1)
                )
                L2 = (
                    L1_0+L2_0+L3_0
                    +(m*wl/neff/2)*(i-1)
                    -L1-L3
                )

                x1 = ports_i[i-1][0] + L1*np.cos(rotation_theta)
                y1 = ports_i[i-1][1] + L1*np.sin(rotation_theta)

                xc1 = x1 - Re*np.sin(rotation_theta)
                yc1 = y1 + Re*np.cos(rotation_theta)
                x2 = xc1 + Re
                y2 = yc1
                angle_start = rotation_angle-90
                angle_end = 0
            else:
                L1 = (
                    x_center-L3_0-Re-deltaW*(i-1)
                    -ports_i[i-1][0]
                    +Re*(1-np.sin(rotation_theta))
                ) / np.cos(rotation_theta)
                L3 = (
                    L3_0
                    +deltaW*(i-1)
                )
                L2 = (
                    L1_0+L2_0+L3_0
                    +(m*wl/neff/2)*(i-1)
                    -L1-L3
                )

                x1 = ports_i[i-1][0] + L1*np.cos(rotation_theta)
                y1 = ports_i[i-1][1] + L1*np.sin(rotation_theta)

                xc1 = x1 + Re*np.sin(rotation_theta)
                yc1 = y1 - Re*np.cos(rotation_theta)
                x2 = xc1 - Re
                y2 = yc1
                angle_start = 180
                angle_end = 90+rotation_angle
            
            x3 = x2
            y3 = y2 + L2
            xc2 = x3 + Re
            yc2 = y3
            x4 = xc2
            y4 = yc2 + Re
            x5 = x4 + L3
            y5 = y4

        # No.1 straight waveguide
        wg1_left = waveguide(
            start=(ports_i[i-1][0], ports_i[i-1][1]),
            end=(x1, y1),
            width=wa,
            cross_section=cross_section
        )
        c.add_ref(wg1_left)
        wg1_right = waveguide(
            start=(2*x_center-ports_i[i-1][0], ports_i[i-1][1]),
            end=(2*x_center-x1, y1),
            width=wa,
            cross_section=cross_section
        )
        c.add_ref(wg1_right)

        # No.1 euler bend waveguide
        bend1_left = bend_euler(
            radius=Re,
            width=wa,
            angle_start=angle_start,
            angle_end=angle_end,
            center=(xc1, yc1),
            cross_section=cross_section
        )
        c.add_ref_off_grid(bend1_left)
        bend1_right = bend_euler(
            radius=Re,
            width=wa,
            angle_start=180-angle_end,
            angle_end=180-angle_start,
            center=(2*x_center-xc1, yc1),
            cross_section=cross_section
        )
        c.add_ref_off_grid(bend1_right)

        # No.2 straight waveguide
        wg2_left = waveguide(
            start=(x2, y2),
            end=(x3, y3),
            width=wa,
            cross_section=cross_section
        )
        c.add_ref(wg2_left)
        wg2_right = waveguide(
            start=(2*x_center-x2, y2),
            end=(2*x_center-x3, y3),
            width=wa,
            cross_section=cross_section
        )
        c.add_ref(wg2_right)
        
        # No.2 euler bend waveguide
        bend2_left = bend_euler(
            radius=Re,
            width=wa,
            angle_start=90,
            angle_end=180,
            center=(xc2, yc2),
            cross_section=cross_section
        )
        c.add_ref_off_grid(bend2_left)
        bend2_right = bend_euler(
            radius=Re,
            width=wa,
            angle_start=0,
            angle_end=90,
            center=(2*x_center-xc2, yc2),
            cross_section=cross_section
        )
        c.add_ref_off_grid(bend2_right)

        # No.3 straight waveguide
        wg3_left = waveguide(
            start=(x4, y4),
            end=(x5, y5),
            width=wa,
            cross_section=cross_section
        )
        c.add_ref(wg3_left)
        wg3_right = waveguide(
            start=(2*x_center-x4, y4),
            end=(2*x_center-x5, y5),
            width=wa,
            cross_section=cross_section
        )
        c.add_ref(wg3_right)

    # Output FPR (right)
    FPR_o, _ = fpr(
        inputs=No,
        outputs=Na,
        w0=w0,
        wi=wo,
        wa=wa,
        di=do,
        do=da,
        gap=gap,
        lti=lto,
        lto=lta,
        mi=mo,
        mo=ma,
        ls=ls,
        ltt=ltt,
        radius=Rm,
        Ra=Ra,
        center=(2*x_center, 0),
        rotation_angle=180-rotation_angle,
        cross_section=cross_section,
        scale=1.5
    )
    c.add_ref(FPR_o)

    return c

if __name__ == "__main__":
    AWG = awg()
    AWG.show()
