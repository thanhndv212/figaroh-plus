"""FIGAROH mark, wordmark and README banner.

Regenerates docs/source/assets/:

    python3 docs/development/make_logo.py docs/source/assets
"""

import math
import os
import sys

PALETTES = {
    "light": dict(
        ink="#1C2541",
        leg="#0B132B",
        brass="#C8A24A",
        tick="#8D99C4",
        text="#1C2541",
        sub="#5C6789",
    ),
    "dark": dict(
        ink="#D5DBEA",
        leg="#F2F4F8",
        brass="#D4B062",
        tick="#1C2541",
        text="#EEF1F7",
        sub="#9AA5C8",
    ),
}

# Mark geometry (512 grid, 8 px module)
X0, X1, Y0, Y1, W = 144, 384, 96, 416, 56  # carpenter's square
HX, HY = X0 + W / 2, (Y0 + Y1) / 2  # hinge: stem centreline, vertical centre
L, HALF = 160, 5.0  # compass leg length, half opening (deg)
R = L  # arc passes through both tips


def f(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def polar(r, a):
    a = math.radians(a)
    return HX + r * math.cos(a), HY + r * math.sin(a)


def points(ps):
    return " ".join(f"{f(x)},{f(y)}" for x, y in ps)


def leg(sign):
    a = sign * HALF
    ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
    nx, ny = -uy, ux
    ex, ey = polar(L - 22, a)
    tip = polar(L, a)
    pts = [
        (HX + 10 * nx, HY + 10 * ny),
        (ex + 3.5 * nx, ey + 3.5 * ny),
        tip,
        (ex - 3.5 * nx, ey - 3.5 * ny),
        (HX - 10 * nx, HY - 10 * ny),
    ]
    nib = [(ex - 3.5 * nx, ey - 3.5 * ny), tip, (ex + 3.5 * nx, ey + 3.5 * ny)]
    return points(pts), points(nib)


def mark(p):
    """The mark in 512-grid coordinates (content box x 144..384, y 96..416)."""
    s = []
    a0, a1 = polar(R, -30), polar(R, 30)
    s.append(
        f'<path d="M {f(a0[0])} {f(a0[1])} A {R} {R} 0 0 1 {f(a1[0])} {f(a1[1])}" '
        f'fill="none" stroke="{p["brass"]}" stroke-width="2"/>'
    )
    s.append(f'<g stroke="{p["brass"]}" stroke-width="2" stroke-linecap="round">')
    for d in range(-30, 31, 10):
        q0, q1 = polar(R, d), polar(R + (10 if d % 30 == 0 else 6), d)
        s.append(
            f'<line x1="{f(q0[0])}" y1="{f(q0[1])}" x2="{f(q1[0])}" y2="{f(q1[1])}"/>'
        )
    s.append("</g>")
    s.append(
        f'<path d="M{X0} {Y0} H{X1} V{Y0 + W} H{X0 + W} V{Y1} H{X0} Z" fill="{p["ink"]}"/>'
    )
    s.append(
        f'<g stroke="{p["tick"]}" stroke-width="2.5" stroke-linecap="square" opacity="0.9">'
    )
    for y in range(Y0 + 64, Y1 - 7, 16):
        if abs(y - HY) < 48:
            continue
        ln = 16 if (y - Y0) % 64 == 0 else 8
        s.append(f'<line x1="{X0 + 8}" y1="{y}" x2="{X0 + 8 + ln}" y2="{y}"/>')
    for x in range(X0 + 64, X1 - 7, 16):
        ln = 16 if (x - X0) % 64 == 0 else 8
        s.append(f'<line x1="{x}" y1="{Y0 + 8}" x2="{x}" y2="{Y0 + 8 + ln}"/>')
    s.append("</g>")
    for sign in (-1, 1):
        pts, nib = leg(sign)
        s.append(f'<polygon points="{pts}" fill="{p["leg"]}"/>')
        if sign == 1:
            s.append(f'<polygon points="{nib}" fill="{p["brass"]}"/>')
    s.append(
        f'<circle cx="{f(HX)}" cy="{f(HY)}" r="22" fill="{p["leg"]}" '
        f'stroke="{p["brass"]}" stroke-width="2"/>'
    )
    s.append(f'<circle cx="{f(HX)}" cy="{f(HY)}" r="6" fill="{p["brass"]}"/>')
    return "\n".join(s)


# Wordmark: monoline geometric capitals, cap height 96, stroke 16, 8 px module.
CAP, SW, GAP = 96, 16, 24
h = SW / 2
LETTERS = {  # (advance, centre-line paths, overshooting round?)
    "F": (64, [f"M{h} 0 V{CAP}", f"M0 {h} H64", "M0 48 H52"]),
    "I": (16, [f"M{h} 0 V{CAP}"]),
    "G": (96, [None]),  # built below
    "A": (80, [f"M-2 {CAP + 16} L40 -24 L82 {CAP + 16}", "M16 68 H64"]),
    "R": (
        72,
        [f"M{h} {CAP} V{h} H40 A20 20 0 0 1 40 48 H{h}", f"M36 48 L76 {CAP + 16}"],
    ),
    "O": (96, [None]),
    "H": (80, [f"M{h} 0 V{CAP}", f"M{80 - h} 0 V{CAP}", "M0 48 H80"]),
}


def round_paths(ch):
    r = 48 - h + 1  # 1 px optical overshoot above and below the cap band
    if ch == "O":
        return [f'<circle cx="48" cy="48" r="{f(r)}" fill="none"/>']
    a = math.radians(-40)
    sx, sy = 48 + r * math.cos(a), 48 + r * math.sin(a)
    return [
        f'<path d="M{f(sx)} {f(sy)} A{f(r)} {f(r)} 0 1 0 {f(48 + r)} 48 H52" fill="none"/>'
    ]


def wordmark(p, x0, y0, clip_id):
    s = [
        f'<clipPath id="{clip_id}"><rect x="-8" y="-1" width="2000" height="{CAP + 2}"/></clipPath>',
        f'<g transform="translate({x0} {y0})" stroke="{p["text"]}" stroke-width="{SW}" '
        f'stroke-linecap="butt" stroke-linejoin="miter" fill="none" clip-path="url(#{clip_id})">',
    ]
    x = 0
    for ch in "FIGAROH":
        adv, paths = LETTERS[ch]
        s.append(f'<g transform="translate({x} 0)">')
        if paths == [None]:
            s += round_paths(ch)
        else:
            s += [f'<path d="{d}"/>' for d in paths]
        s.append("</g>")
        x += adv + GAP
    s.append("</g>")
    return "\n".join(s), x - GAP


def svg(w, hgt, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {hgt}" width="{w}" '
        f'height="{hgt}" role="img" aria-labelledby="t">\n<title id="t">{title}</title>\n'
        f"{body}\n</svg>\n"
    )


def banner(p, name):
    W_, H_ = 1280, 320
    k = 0.6  # mark scale: 320 -> 192 px tall
    mw = (X1 - X0) * k
    wm, wm_w = wordmark(p, 0, 0, f"cap-{name}")
    total = mw + 64 + wm_w
    mx = (W_ - total) / 2
    top = (H_ - (Y1 - Y0) * k) / 2  # 64
    axis = top + (HY - Y0) * k  # compass axis, 160
    wx = mx + mw + 64
    body = (
        [f'<rect width="{W_}" height="{H_}" rx="16" fill="#FFFFFF"/>']
        if name == "light"
        else []
    )
    body += [
        f'<g transform="translate({f(mx - X0 * k)} {f(top - Y0 * k)}) scale({k})">',
        mark(p),
        "</g>",
    ]
    # wordmark: cap band from the mark's top to the compass axis
    wm, _ = wordmark(p, f(wx), f(top), f"cap-{name}")
    body.append(wm)
    # ruler hairline under the wordmark, graduated every 16 px
    ry = axis + 24
    body.append(f'<g stroke="{p["brass"]}" stroke-width="2" stroke-linecap="square">')
    body.append(f'<line x1="{f(wx)}" y1="{f(ry)}" x2="{f(wx + wm_w)}" y2="{f(ry)}"/>')
    i = 0
    xx = wx
    while xx <= wx + wm_w + 0.01:
        ln = 10 if i % 4 == 0 else 5
        body.append(f'<line x1="{f(xx)}" y1="{f(ry)}" x2="{f(xx)}" y2="{f(ry - ln)}"/>')
        xx += 16
        i += 1
    body.append("</g>")
    body.append(
        f'<text x="{f(wx)}" y="{f(top + (Y1 - Y0) * k - 4)}" fill="{p["sub"]}" '
        "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\" "
        f'font-size="19" font-weight="500" letter-spacing="3.2" textLength="{f(wm_w)}" '
        'lengthAdjust="spacing">GEOMETRIC CALIBRATION · DYNAMIC IDENTIFICATION</text>'
    )
    return svg(W_, H_, "\n".join(body), "FIGAROH")


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, p in PALETTES.items():
        with open(os.path.join(outdir, f"figaroh-mark-{name}.svg"), "w") as fh:
            fh.write(
                svg(
                    512,
                    512,
                    f'<g transform="translate(-8 0)">\n{mark(p)}\n</g>',
                    "FIGAROH",
                )
            )
        if name == "light":
            with open(os.path.join(outdir, "figaroh-banner.svg"), "w") as fh:
                fh.write(banner(p, name))


if __name__ == "__main__":
    main(sys.argv[1])
