"""
ui_art.py - hand-drawn SVG art for the Streamlit app: team shield badges, the hero stadium scene,
trophy, bat, ball and stumps. No external images.

Logos: these are original badge drawings in each team's colours - NOT the official logos (those are trademarks
of the franchises). To use official logos you are licensed to use, drop PNG files named by team code
(e.g. assets/logos/CSK.png) into assets/logos/ and the app will show them instead of the drawn badges.
"""
import base64
import random
from pathlib import Path

LOGO_DIR = Path("assets/logos")
_TW = ' class="tw"'

TEAMS = {
    "Chennai Super Kings":         {"code": "CSK",  "c1": "#F9CD05", "c2": "#E58E00", "ink": "#0B2A6F"},
    "Delhi Capitals":              {"code": "DC",   "c1": "#17479E", "c2": "#EF1B23", "ink": "#FFFFFF"},
    "Gujarat Titans":              {"code": "GT",   "c1": "#1C2A4A", "c2": "#B28E4F", "ink": "#F5E6C4"},
    "Kolkata Knight Riders":       {"code": "KKR",  "c1": "#3A225D", "c2": "#B8860B", "ink": "#F7D774"},
    "Lucknow Super Giants":        {"code": "LSG",  "c1": "#0B6FC2", "c2": "#F58220", "ink": "#FFFFFF"},
    "Mumbai Indians":              {"code": "MI",   "c1": "#004BA0", "c2": "#2E86DE", "ink": "#F2C14E"},
    "Punjab Kings":                {"code": "PBKS", "c1": "#D71920", "c2": "#8E0E14", "ink": "#F1F1F1"},
    "Rajasthan Royals":            {"code": "RR",   "c1": "#EA1A85", "c2": "#254AA5", "ink": "#FFFFFF"},
    "Royal Challengers Bengaluru": {"code": "RCB",  "c1": "#D1171B", "c2": "#1B1B1B", "ink": "#F6C945"},
    "Sunrisers Hyderabad":         {"code": "SRH",  "c1": "#F26522", "c2": "#2B2B2B", "ink": "#FFFFFF"},
}


def team(name):
    return TEAMS.get(name, {"code": (name[:3] or "?").upper(), "c1": "#44506B", "c2": "#222B3D", "ink": "#FFFFFF"})


def badge(name, size=56):
    """Shield badge. Uses assets/logos/<CODE>.png if present."""
    t = team(name)
    png = LOGO_DIR / f"{t['code']}.png"
    if png.exists():
        b64 = base64.b64encode(png.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" style="object-fit:contain" alt="{t["code"]}"/>'
    code = t["code"]
    fs = 30 if len(code) <= 2 else 26 if len(code) == 3 else 21
    gid = f"g{code}{size}"
    return (
        f'<svg width="{size}" height="{round(size * 1.2)}" viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{name}">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{t["c1"]}"/><stop offset="1" stop-color="{t["c2"]}"/></linearGradient>'
        f'<linearGradient id="{gid}s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".45"/><stop offset=".55" stop-color="#fff" stop-opacity="0"/></linearGradient></defs>'
        f'<path d="M50 3 L93 17 V58 C93 89 73 109 50 117 C27 109 7 89 7 58 V17 Z" fill="#0a0f24" opacity=".55" transform="translate(0,3)"/>'
        f'<path d="M50 3 L93 17 V58 C93 89 73 109 50 117 C27 109 7 89 7 58 V17 Z" fill="url(#{gid})" stroke="#F9C74F" stroke-width="3.5"/>'
        f'<path d="M50 11 L85 22 V58 C85 84 68 101 50 108 C32 101 15 84 15 58 V22 Z" fill="none" stroke="{t["ink"]}" stroke-opacity=".55" stroke-width="1.6"/>'
        f'<path d="M50 3 L93 17 V50 C70 40 30 40 7 50 V17 Z" fill="url(#{gid}s)"/>'
        f'<text x="50" y="{66 + (fs - 26) // 3}" text-anchor="middle" font-family="Bebas Neue, Impact, Arial Black, sans-serif" '
        f'font-size="{fs + 8}" font-weight="900" letter-spacing="1" fill="{t["ink"]}" stroke="#000" stroke-opacity=".25" stroke-width=".8">{code}</text>'
        f'<circle cx="50" cy="90" r="5.5" fill="#C1272D"/><path d="M45 88.5 q5 3 10 0 M45 91.5 q5 -3 10 0" stroke="#fff" stroke-width="1" fill="none"/>'
        f'</svg>')


# ------------------------------------------------------------------ pieces ---------------
def trophy(cx=0, cy=0, s=1.0):
    return (
        f'<g transform="translate({cx},{cy}) scale({s})">'
        '<path d="M-42,-70 h84 c0,46 -16,78 -42,90 c-26,-12 -42,-44 -42,-90z" fill="url(#gold)" stroke="#8a5a00" stroke-width="2"/>'
        '<path d="M-42,-58 c-30,0 -36,34 -6,46" fill="none" stroke="url(#gold)" stroke-width="9" stroke-linecap="round"/>'
        '<path d="M42,-58 c30,0 36,34 6,46" fill="none" stroke="url(#gold)" stroke-width="9" stroke-linecap="round"/>'
        '<rect x="-9" y="18" width="18" height="34" fill="url(#gold)"/>'
        '<rect x="-34" y="50" width="68" height="14" rx="4" fill="url(#gold)" stroke="#8a5a00" stroke-width="2"/>'
        '<rect x="-44" y="62" width="88" height="14" rx="4" fill="#3b2a0b"/>'
        '<path d="M-30,-62 c-4,30 2,54 22,70" fill="none" stroke="#fff" stroke-opacity=".55" stroke-width="6" stroke-linecap="round"/>'
        '<circle cx="0" cy="-30" r="13" fill="#fff" fill-opacity=".18"/>'
        '</g>')


def bat(cx=0, cy=0, rot=-32, s=1.0):
    return (
        f'<g transform="translate({cx},{cy}) rotate({rot}) scale({s})">'
        '<rect x="-15" y="-92" width="30" height="108" rx="7" fill="url(#wood)" stroke="#7a5428" stroke-width="2"/>'
        '<path d="M0,-92 V16" stroke="#7a5428" stroke-opacity=".5" stroke-width="2"/>'
        '<rect x="-5" y="-134" width="10" height="46" rx="4" fill="#1b1b2f"/>'
        '<path d="M-5,-124 h10 M-5,-116 h10 M-5,-108 h10 M-5,-100 h10" stroke="#F9C74F" stroke-width="1.6"/>'
        '</g>')


def ball(cx=0, cy=0, r=16):
    return (f'<g transform="translate({cx},{cy})"><circle r="{r}" fill="url(#redball)" stroke="#5c0a0d" stroke-width="1.5"/>'
            f'<path d="M{-r*0.75:.1f},{-r*0.45:.1f} q{r*0.75:.1f},{r*0.45:.1f} {r*1.5:.1f},0 M{-r*0.75:.1f},{r*0.45:.1f} q{r*0.75:.1f},{-r*0.45:.1f} {r*1.5:.1f},0" '
            'stroke="#fff" stroke-width="1.6" stroke-dasharray="2.4 2" fill="none"/></g>')


def stumps(cx=0, cy=0, s=1.0):
    return (
        f'<g transform="translate({cx},{cy}) scale({s})">'
        + "".join(f'<rect x="{x - 3}" y="-70" width="6" height="70" rx="2" fill="#F3E3C0" stroke="#a98b55" stroke-width="1"/>' for x in (-16, 0, 16))
        + '<rect x="-19" y="-74" width="14" height="4" rx="2" fill="#E04F39"/><rect x="-3" y="-74" width="14" height="4" rx="2" fill="#E04F39"/>'
        '</g>')


DEFS = (
    '<defs>'
    '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#040918"/><stop offset=".55" stop-color="#0B1F4B"/><stop offset="1" stop-color="#27227A"/></linearGradient>'
    '<radialGradient id="glow" cx=".5" cy="1" r=".75"><stop offset="0" stop-color="#FF8A00" stop-opacity=".85"/><stop offset=".45" stop-color="#FF5E00" stop-opacity=".25"/><stop offset="1" stop-color="#FF5E00" stop-opacity="0"/></radialGradient>'
    '<linearGradient id="gold" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFF3B0"/><stop offset=".4" stop-color="#F9C74F"/><stop offset="1" stop-color="#C98A0B"/></linearGradient>'
    '<linearGradient id="wood" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#F0CF9B"/><stop offset="1" stop-color="#C89B5E"/></linearGradient>'
    '<radialGradient id="redball" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#FF6B6B"/><stop offset="1" stop-color="#B3121A"/></radialGradient>'
    '<linearGradient id="cone" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFF6C8" stop-opacity=".55"/><stop offset="1" stop-color="#FFF6C8" stop-opacity="0"/></linearGradient>'
    '<linearGradient id="bowl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#101A45"/><stop offset="1" stop-color="#070C22"/></linearGradient>'
    '</defs>')


def _tower(x, flip=1):
    lights = "".join(f'<circle cx="{x + flip * (dx)}" cy="{y}" r="4.2" fill="#FFF8D6"/>' for y in (72, 84, 96) for dx in (-16, -5, 6, 17))
    return (f'<rect x="{x - 4}" y="96" width="8" height="190" fill="#0a1030"/>'
            f'<rect x="{x - 26}" y="62" width="52" height="46" rx="4" fill="#161f52" stroke="#2b3a86"/>' + lights +
            f'<polygon points="{x - 24},108 {x + 24},108 {x + 190 * flip + 60},300 {x + 190 * flip - 140},300" fill="url(#cone)" class="cone"/>')


def hero_svg(width=1200, height=300):
    rnd = random.Random(2027)
    stars = "".join(f'<circle cx="{rnd.randint(0, 1200)}" cy="{rnd.randint(0, 150)}" r="{rnd.choice([.8, 1, 1.4])}" fill="#fff" opacity="{rnd.uniform(.3, .9):.2f}"'
                    f'{_TW if rnd.random() < .25 else ""}/>' for _ in range(70))
    crowd = []
    for row, (y0, amp) in enumerate(((226, 22), (240, 20), (254, 18), (268, 16))):
        for i in range(0, 1200, 7):
            y = y0 - amp * (1 - ((i - 600) / 600) ** 2) + rnd.uniform(-2, 2)
            crowd.append(f'<circle cx="{i + rnd.uniform(-2, 2):.1f}" cy="{y:.1f}" r="{rnd.uniform(1.6, 2.6):.1f}" '
                         f'fill="{rnd.choice(["#FF8A00", "#F9C74F", "#4EA1FF", "#fff", "#E8344E", "#9B7BFF"])}" opacity="{rnd.uniform(.35, .85):.2f}"/>')
    return (
        f'<svg viewBox="0 0 1200 300" width="100%" height="{height}" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">'
        + DEFS +
        '<rect width="1200" height="300" fill="url(#sky)"/><rect width="1200" height="300" fill="url(#glow)"/>' + stars +
        _tower(120, 1) + _tower(1080, -1) +
        '<path d="M-20,300 V228 Q600,150 1220,228 V300Z" fill="url(#bowl)"/>' + "".join(crowd) +
        '<ellipse cx="600" cy="292" rx="330" ry="22" fill="#2E8B57" opacity=".55"/><ellipse cx="600" cy="292" rx="150" ry="9" fill="#C9E7A9" opacity=".5"/>'
        + stumps(470, 292, 0.85) + bat(520, 300, -28, 0.75) + ball(560, 280, 11) +
        '<g class="trophy">' + trophy(995, 172, 1.0) + '</g>'
        '<g opacity=".9">' + ball(250, 70, 9) + ball(1010, 56, 7) + '</g>'
        '</svg>')


def small_icons():
    """Bat + ball + stumps strip used as a section divider."""
    return ('<svg viewBox="0 0 260 60" width="190" height="44" xmlns="http://www.w3.org/2000/svg">' + DEFS +
            stumps(40, 58, .62) + bat(110, 66, -48, .42) + ball(165, 40, 9) + trophy(226, 30, .3) + '</svg>')


def hero_title_svg():
    """Two-line logo lockup drawn inside an SVG layer (scales with the banner, immune to page CSS)."""
    return (
        '<svg viewBox="0 0 1200 300" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
        'style="position:absolute;left:0;top:0;width:100%;height:100%">'
        '<defs><linearGradient id="ttl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/>'
        '<stop offset=".55" stop-color="#F9C74F"/><stop offset="1" stop-color="#FF8A00"/></linearGradient>'
        '<filter id="tglow" x="-10%" y="-30%" width="120%" height="160%"><feGaussianBlur stdDeviation="5" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        '<text x="600" y="132" text-anchor="middle" font-family="Bebas Neue, Impact, Arial Black, sans-serif" '
        'font-size="150" textLength="440" lengthAdjust="spacingAndGlyphs" fill="url(#ttl)" filter="url(#tglow)">IPL 2027</text>'
        '<text x="600" y="194" text-anchor="middle" font-family="Bebas Neue, Impact, Arial Black, sans-serif" '
        'font-size="68" textLength="440" lengthAdjust="spacingAndGlyphs" fill="#F9C74F" stroke="#000" stroke-opacity=".3" '
        'stroke-width="1">TITLE PREDICTOR</text></svg>')
