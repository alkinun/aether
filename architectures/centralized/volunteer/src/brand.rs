//! Branding: a restrained palette for the launcher UI.
//!
//! The identity is built around a small burnt-neon palette: rose, smoked cyan,
//! ember amber, ash violet, and warm bone. Semantic colors intentionally reuse
//! those same tones so the launcher never drifts into unrelated green/red/yellow.
//! There is no
//! animated rainbow gradient by design — text is rendered in solid colors and
//! the only motion in the UI is the build spinner / progress sweep.

use ratatui::style::Color;

// --- Palette ---------------------------------------------------------------
pub const BRAND_A: Color = Color::Rgb(218, 78, 138); // burnt rose (primary)
pub const BRAND_B: Color = Color::Rgb(82, 184, 205); // smoked cyan (secondary)
pub const ACCENT_AMBER: Color = Color::Rgb(226, 136, 68); // ember amber
pub const ACCENT_VIOLET: Color = Color::Rgb(168, 92, 188); // ash violet
pub const BLOOM_BONE: Color = Color::Rgb(226, 204, 184); // warm bloom/bone
pub const BASE_EMBER: Color = Color::Rgb(72, 56, 56);

pub const SUCCESS: Color = BRAND_B;
pub const WARN: Color = ACCENT_AMBER;
pub const DANGER: Color = BRAND_A;
pub const DIM: Color = Color::Rgb(116, 98, 104);
pub const INK: Color = BLOOM_BONE;
pub const PANEL: Color = Color::Rgb(20, 18, 22);
pub const PANEL_HI: Color = Color::Rgb(70, 56, 64);

pub fn rgb(c: Color) -> (f32, f32, f32) {
    match c {
        Color::Rgb(r, g, b) => (r as f32, g as f32, b as f32),
        _ => (200.0, 200.0, 200.0),
    }
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let (ar, ag, ab) = rgb(a);
    let (br, bg, bb) = rgb(b);
    Color::Rgb(
        lerp(ar, br, t) as u8,
        lerp(ag, bg, t) as u8,
        lerp(ab, bb, t) as u8,
    )
}
