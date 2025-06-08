import math

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    d = float(trip_duration_days)
    m = float(miles_traveled)
    r = float(total_receipts_amount)

    # Safe denominators
    d_safe = d + 1e-6
    m_safe = m + 1e-6

    # Basic powers
    d2, d3 = d * d, d * d * d
    m2, m3 = m * m, m * m * m
    r2, r3 = r * r, r * r * r

    # Interaction terms
    dm = d * m
    dr = d * r
    mr = m * r
    d2m = d2 * m
    d2r = d2 * r
    dm2 = d * m2
    dmr = d * m * r
    dr2 = d * r2
    m2r = m2 * r
    mr2 = m * r2

    # Overspend features
    over = max(0.0, r - 800.0)
    over2 = over * over
    over3 = over2 * over
    overd = over * d
    overm = over * m
    overr = over * r

    # Ratio / inverse / log features
    rpd = r / d_safe
    rpp = r / m_safe
    mpd = m / d_safe
    invd = 1.0 / d_safe
    invm = 1.0 / m_safe
    logr = math.log1p(r)
    logm = math.log1p(m)

    # Lux features
    lux = 1.0 if r > 1500 else 0.0
    lux2 = 1.0 if rpd > 200 else 0.0

    # Coefficients updated (λ=10 rich model)
    coef = [
        -11.386792757689308,
        128.89549765797815,
        -0.0007983363033925145,
        -0.048630556943350244,
        -11.71211940103739,
        0.00020061783813348064,
        9.40043476273271e-05,
        0.08112397945800302,
        -0.01652700945628891,
        0.00017999059205736097,
        0.5069754486629876,
        -3.082549851801814e-07,
        6.823601722543471e-07,
        -0.007242896865931405,
        0.0011033769473505434,
        4.826290938483638e-05,
        -9.790994447210301e-06,
        -3.865764489223714e-06,
        3.941696637919952e-09,
        1.9368767208749147e-07,
        5.2553897295325635e-06,
        -0.003535755923835485,
        -2.903254138218345e-07,
        0.015556530320583463,
        -0.0009767140941685515,
        0.0002725579227967527,
        -0.0674767334664814,
        0.12560778981257684,
        0.2513225418988455,
        -42.657511470211375,
        -0.8495096208438094,
        6.742532635558535,
        16.222826587066205,
        -0.0009118979000005852,
        5.626368106275872,
        -0.25417230574243793,
        0.475346258763239,
        -11.325919602363001,
        -23.592639867402337,
        0.04009529464968733,
        0.13852743233186837,
    ]

    # Feature vector in same order
    features = [
        1.0,
        d,
        m,
        r,
        d2,
        m2,
        r2,
        dm,
        dr,
        mr,
        d3,
        m3,
        r3,
        d2m,
        d2r,
        dm2,
        dmr,
        dr2,
        m2r,
        mr2,
        over,
        over2,
        over3,
        overd,
        overm,
        overr,
        rpd,
        rpp,
        mpd,
        invd,
        invm,
        logr,
        logm,
        lux,
        lux*d,
        lux*r,
        lux*over,
        lux2,
        lux2*d,
        lux2*r,
        lux2*over,
    ]

    total = sum(c * f for c, f in zip(coef, features))
    return round(total, 2) 