//! Terminal adaptation of the Aether Compute glitch logo.
//!
//! The original HTML version layers cyan/magenta channels, character glow,
//! corruption, line jolts, and tears. This keeps the same language but confines
//! it to the logo area so the rest of the launcher remains calm.

use crate::brand;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
};

const LOGO: &str = include_str!("logo.txt");
const JUNK: &[char] = &['@', '#', '$', '%', '&', '?', '!', '<', '>', '{', '}', '[', ']'];

struct LogoCanvas {
    area: Rect,
    origin_x: u16,
    origin_y: u16,
}

pub fn width() -> u16 {
    lines()
        .iter()
        .map(|l| l.chars().count() as u16)
        .max()
        .unwrap_or(0)
}

pub fn height() -> u16 {
    lines().len() as u16
}

pub fn lines() -> Vec<&'static str> {
    LOGO.lines().collect()
}

pub fn draw(buf: &mut Buffer, area: Rect, frame: u64) {
    let lines = lines();
    let logo_w = width();
    let logo_h = lines.len() as u16;
    if logo_w == 0 || logo_h == 0 || area.width < logo_w || area.height < logo_h {
        return;
    }

    let origin_x = area.x + area.width.saturating_sub(logo_w) / 2;
    let origin_y = area.y + area.height.saturating_sub(logo_h) / 2;

    let canvas = LogoCanvas {
        area,
        origin_x,
        origin_y,
    };

    for (row, line) in lines.iter().enumerate() {
        let y = canvas.origin_y + row as u16;
        if flicker_hidden_line(frame, row as u32) {
            continue;
        }
        let x_shift = line_shift(frame, row as u32);
        draw_line(buf, canvas.area, canvas.origin_x, y, x_shift, line, base_style(frame));
    }

    draw_channels(buf, &canvas, &lines, frame);
    draw_corruption(buf, area, origin_x, origin_y, &lines, frame);
    draw_glints(buf, area, origin_x, origin_y, &lines, frame);
}

fn base_style(frame: u64) -> Style {
    let pulse = ((frame as f32 * 0.045).sin() + 1.0) * 0.5;
    let col = brand::lerp_color(brand::BASE_EMBER, brand::BLOOM_BONE, 0.28 + pulse * 0.10);
    Style::default().fg(col)
}

fn draw_line(
    buf: &mut Buffer,
    area: Rect,
    origin_x: u16,
    y: u16,
    x_shift: i16,
    line: &str,
    style: Style,
) {
    let x = shifted_x(origin_x, x_shift);
    if y < area.y || y >= area.y + area.height {
        return;
    }
    for (col, ch) in line.chars().enumerate() {
        let x = x + col as u16;
        if x >= area.x + area.width {
            break;
        }
        if x >= area.x {
            buf[(x, y)].set_char(ch).set_style(style);
        }
    }
}

fn draw_channels(buf: &mut Buffer, canvas: &LogoCanvas, lines: &[&str], frame: u64) {
    let Some(burst) = burst_frame(frame) else {
        return;
    };
    let seed = hash((frame / 37) as u32 ^ 0x4d2f);
    let count = (seed.wrapping_rem(2) + 1) as usize;
    for channel in 0..count {
        let ch_seed = hash(seed ^ (channel as u32).wrapping_mul(0x9e37));
        let color = glitch_color(ch_seed);
        let dir = if ch_seed.is_multiple_of(2) { 1 } else { -1 };
        draw_channel(buf, canvas, lines, frame + burst + channel as u64 * 11, color, dir);
    }
}

fn draw_channel(
    buf: &mut Buffer,
    canvas: &LogoCanvas,
    lines: &[&str],
    frame: u64,
    color: Color,
    dir: i16,
) {
    let start = (hash(frame as u32).wrapping_rem(lines.len() as u32)) as usize;
    let band_h = (hash(frame as u32 ^ 0x8a13).wrapping_rem(6) + 1) as usize;
    let shift = dir * ((hash(frame as u32 ^ 0x4f2d).wrapping_rem(6) + 1) as i16);
    let style = Style::default().fg(color).add_modifier(Modifier::BOLD);

    for (row, line) in lines.iter().enumerate().skip(start).take(band_h) {
        let y = canvas.origin_y + row as u16;
        draw_line(buf, canvas.area, canvas.origin_x, y, shift, line, style);
    }
}

fn draw_corruption(buf: &mut Buffer, area: Rect, origin_x: u16, origin_y: u16, lines: &[&str], frame: u64) {
    if !frame.is_multiple_of(7 + (hash((frame / 23) as u32) % 9) as u64) {
        return;
    }
    let count = (hash(frame as u32 ^ 0x73f1).wrapping_rem(7) + 1) as usize;
    for i in 0..count {
        let seed = hash(frame as u32 ^ (i as u32).wrapping_mul(0x9e37));
        let row = seed.wrapping_rem(lines.len() as u32) as usize;
        let line_w = lines[row].chars().count().max(1) as u32;
        let col = hash(seed ^ 0x41).wrapping_rem(line_w) as u16;
        let x = origin_x + col;
        let y = origin_y + row as u16;
        if x < area.x + area.width && y < area.y + area.height {
            let ch = JUNK[hash(seed ^ 0x91).wrapping_rem(JUNK.len() as u32) as usize];
            buf[(x, y)].set_char(ch).set_style(
                Style::default()
                    .fg(glitch_color(seed ^ 0xa17))
                    .add_modifier(Modifier::BOLD),
            );
        }
    }
}

fn draw_glints(buf: &mut Buffer, area: Rect, origin_x: u16, origin_y: u16, lines: &[&str], frame: u64) {
    let window = 2 + hash((frame / 31) as u32).wrapping_rem(3) as u64;
    if frame % 19 > window {
        return;
    }
    let count = (hash(frame as u32 ^ 0x5bd1).wrapping_rem(8) + 2) as usize;
    for i in 0..count {
        let seed = hash(frame as u32 ^ (i as u32).wrapping_mul(0x85eb));
        let row = seed.wrapping_rem(lines.len() as u32) as usize;
        let line_w = lines[row].chars().count().max(1) as u32;
        let col = hash(seed ^ 0xb7).wrapping_rem(line_w) as u16;
        let x = origin_x + col;
        let y = origin_y + row as u16;
        if x < area.x + area.width && y < area.y + area.height {
            let color = glitch_color(seed ^ 0xc2);
            let ch = lines[row].chars().nth(col as usize).unwrap_or(' ');
            buf[(x, y)]
                .set_char(ch)
                .set_style(Style::default().fg(color).add_modifier(Modifier::BOLD));
        }
    }
}

fn burst_frame(frame: u64) -> Option<u64> {
    let period = 62 + hash((frame / 211) as u32).wrapping_rem(57) as u64;
    let local = frame % period;
    let window = 4 + hash((frame / period) as u32 ^ 0x8b57).wrapping_rem(11) as u64;
    (local < window).then_some(local)
}

fn glitch_color(seed: u32) -> Color {
    const COLORS: [Color; 5] = [
        brand::BRAND_B,
        brand::BRAND_A,
        brand::ACCENT_AMBER,
        brand::ACCENT_VIOLET,
        brand::BLOOM_BONE,
    ];
    COLORS[seed.wrapping_rem(COLORS.len() as u32) as usize]
}

fn flicker_hidden_line(frame: u64, row: u32) -> bool {
    let period = 89 + hash((frame / 197) as u32 ^ 0x51a7).wrapping_rem(53) as u64;
    let local = frame % period;
    let window = 2 + hash((frame / period) as u32 ^ 0xa49b).wrapping_rem(5) as u64;
    local < window && hash(row ^ frame as u32 ^ 0x7c15).wrapping_rem(10) < 3
}

fn line_shift(frame: u64, row: u32) -> i16 {
    let period = 67 + hash(row ^ (frame / 173) as u32).wrapping_rem(71) as u64;
    if frame % period > 4 || hash(row ^ frame as u32).wrapping_rem(10) > 3 {
        return 0;
    }
    let mag = (hash(row.wrapping_mul(31) ^ frame as u32).wrapping_rem(7) + 1) as i16;
    if hash(row ^ 0x51f1 ^ frame as u32).is_multiple_of(2) {
        mag
    } else {
        -mag
    }
}

fn shifted_x(x: u16, shift: i16) -> u16 {
    if shift < 0 {
        x.saturating_sub(shift.unsigned_abs())
    } else {
        x.saturating_add(shift as u16)
    }
}

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}
