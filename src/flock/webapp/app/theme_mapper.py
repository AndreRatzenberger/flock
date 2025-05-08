from __future__ import annotations

import re

###########################
#  Low–level colour utils #
###########################

_HEX_RE = re.compile(r"^#?([0-9a-f]{6})$", re.I)

def _hex_to_rgb(hex_str: str) -> tuple[int, int, int] | None:
    m = _HEX_RE.match(hex_str.strip())
    if not m:
        return None
    h = m.group(1)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[misc]

def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def _rel_lum(rgb: tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = map(channel, rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = _rel_lum(a), _rel_lum(b)
    darker, lighter = sorted((la, lb))
    return (lighter + 0.05) / (darker + 0.05)

def _mix(rgb: tuple[int, int, int], other: tuple[int, int, int], pct: float) -> tuple[int, int, int]:
    return tuple(int(rgb[i] * (1 - pct) + other[i] * pct) for i in range(3))

def _rgba(rgb: tuple[int, int, int], alpha: float) -> str:
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

########################################
#  Theme → Pico-CSS conversion engine  #
########################################

AccentOrder = ["blue", "cyan", "magenta", "green", "yellow", "red"]

def _theme_color(theme: dict, group: str, name: str) -> tuple[int, int, int] | None:
    raw = (
        theme.get("colors", {})
             .get(group, {},)
             .get(name)
    )
    return _hex_to_rgb(raw) if raw else None

def _best_contrast(
    candidates: list[tuple[int, int, int]],
    bg: tuple[int, int, int],
    min_ratio: float = 4.5,
) -> tuple[int, int, int]:
    if not candidates:
        raise ValueError("No colours supplied")
    best = max(candidates, key=lambda c: _contrast(c, bg))
    if _contrast(best, bg) >= min_ratio:
        return best
    return best  # return highest available even if below threshold

def alacritty_to_pico(theme: dict) -> dict[str, str]:
    css: dict[str, str] = {}

    # 1  Main surface colours
    bg = _theme_color(theme, "primary", "background") or (0, 0, 0)
    # Try theme's own foreground first, then fall back to palette extremes
    fg_candidates = [
        _theme_color(theme, "primary", "foreground"),
        _theme_color(theme, "normal", "white"),
        _theme_color(theme, "bright", "white"),
        _theme_color(theme, "normal", "black"),
        _theme_color(theme, "bright", "black"),
    ]
    fg = _best_contrast([c for c in fg_candidates if c], bg)

    css["--pico-background-color"] = _rgb_to_hex(bg)
    css["--pico-color"] = _rgb_to_hex(fg)

    # 2  Pick an accent colour that stands out
    accent = None
    for shade in ("normal", "bright"):
        for name in AccentOrder:
            c = _theme_color(theme, shade, name)
            if c and _contrast(c, bg) >= 3:
                accent = c
                break
        if accent:
            break
    accent = accent or fg  # worst case

    accent_hover = _mix(accent, (255, 255, 255), 0.15)              # lighten 15 %
    accent_active = _mix(accent, (0, 0, 0), 0.15)                   # darken 15 %
    accent_focus = _rgba(accent, 0.25)                              # 25 % overlay
    accent_inv = _best_contrast([fg, bg, (255, 255, 255), (0, 0, 0)], accent, 4.5)

    css["--pico-primary"] = _rgb_to_hex(accent)
    css["--pico-primary-hover"] = _rgb_to_hex(accent_hover)
    css["--pico-primary-active"] = _rgb_to_hex(accent_active)
    css["--pico-primary-focus"] = accent_focus
    css["--pico-primary-inverse"] = _rgb_to_hex(accent_inv)

    # 3  Secondary accent = next colour in queue with sufficient contrast
    sec = None
    for name in AccentOrder:
        if sec is None and _hex_to_rgb(css["--pico-primary"]) != _theme_color(theme, "normal", name):
            for shade in ("normal", "bright"):
                c = _theme_color(theme, shade, name)
                if c and _contrast(c, bg) >= 3:
                    sec = c
                    break
    sec = sec or _mix(accent, (255, 255, 255), 0.25)

    css["--pico-secondary"] = _rgb_to_hex(sec)
    css["--pico-secondary-hover"] = _rgb_to_hex(_mix(sec, (255, 255, 255), 0.15))
    css["--pico-secondary-focus"] = _rgba(sec, 0.25)
    css["--pico-secondary-active"] = _rgb_to_hex(sec)
    css["--pico-secondary-inverse"] = _rgb_to_hex(_best_contrast([fg, bg], sec, 4.5))

    # 4  Headings inherit the accent spectrum
    css["--pico-h1-color"] = css["--pico-primary"]
    css["--pico-h2-color"] = css["--pico-secondary"]
    css["--pico-h3-color"] = css["--pico-color"]  # body colour

    # 5  Muted text & borders use the least-contrasty greys we can still read
    grey_candidates = [
        _theme_color(theme, "bright", "black"),
        _theme_color(theme, "normal", "black"),
        _theme_color(theme, "bright", "white"),
        _theme_color(theme, "normal", "white"),
    ]
    muted = _best_contrast([c for c in grey_candidates if c], bg, 3)
    css["--pico-muted-color"] = _rgb_to_hex(muted)
    css["--pico-border-color"] = _rgb_to_hex(_mix(muted, bg, 0.5))
    css["--pico-muted-border-color"] = css["--pico-border-color"]

    # 6  Cards, selections, cursor, code — pick safe defaults
    css["--pico-card-background-color"] = css["--pico-background-color"]
    css["--pico-card-sectioning-background-color"] = _rgb_to_hex(_mix(bg, fg, 0.05))
    css["--pico-card-border-color"] = css["--pico-border-color"]

    sel_bg = _theme_color(theme, "selection", "background") or _mix(bg, fg, 0.20)
    sel_fg = _best_contrast([fg, muted, accent, sec], sel_bg, 4.5)
    css["--pico-selection-background-color"] = _rgb_to_hex(sel_bg)
    css["--pico-selection-color"] = _rgb_to_hex(sel_fg)

    cur_bg = _theme_color(theme, "cursor", "cursor") or sel_bg
    cur_fg = _theme_color(theme, "cursor", "text") or fg
    css["--pico-code-background-color"] = _rgb_to_hex(cur_bg)
    css["--pico-code-color"] = _rgb_to_hex(cur_fg)

    # 7  Form elements and buttons reuse the existing tokens
    css["--pico-form-element-background-color"] = css["--pico-background-color"]
    css["--pico-form-element-border-color"] = css["--pico-border-color"]
    css["--pico-form-element-color"] = css["--pico-color"]
    css["--pico-form-element-focus-color"] = css["--pico-primary-hover"]
    css["--pico-form-element-placeholder-color"] = css["--pico-muted-color"]
    css["--pico-form-element-active-border-color"] = css["--pico-primary"]
    css["--pico-form-element-active-background-color"] = css["--pico-selection-background-color"]
    css["--pico-form-element-disabled-background-color"] = _rgb_to_hex(_mix(bg, fg, 0.1))
    css["--pico-form-element-disabled-border-color"] = css["--pico-border-color"]
    css["--pico-form-element-invalid-border-color"] = _rgb_to_hex(_theme_color(theme, "normal", "red") or accent_active)
    css["--pico-form-element-invalid-focus-color"] = _rgb_to_hex(_theme_color(theme, "bright", "red") or accent_hover)

    # 8  Buttons follow primary palette by default
    css["--pico-button-base-background-color"] = css["--pico-primary"]
    css["--pico-button-base-color"] = css["--pico-primary-inverse"]
    css["--pico-button-hover-background-color"] = css["--pico-primary-hover"]
    css["--pico-button-hover-color"] = css["--pico-primary-inverse"]

    # 9  Semantic markup helpers
    yellow = _theme_color(theme, "normal", "yellow") or _mix(accent, (255, 255, 0), 0.5)
    css["--pico-mark-background-color"] = _rgba(yellow, 0.2)
    css["--pico-mark-color"] = css["--pico-color"]
    css["--pico-ins-color"] = _rgb_to_hex(_theme_color(theme, "normal", "green") or accent)
    css["--pico-del-color"] = _rgb_to_hex(_theme_color(theme, "normal", "red") or accent_active)

    # 10  Contrast helpers
    css["--pico-contrast"] = css["--pico-color"]
    css["--pico-contrast-inverse"] = css["--pico-primary-inverse"]

    return css
