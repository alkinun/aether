//! The onboarding TUI: a small screen state machine driving ratatui rendering.
//!
//! Screens: Welcome -> Form -> Identity -> Build -> Ready (then exec).
//! All rendering is hand-painted with the brand gradient for a consistent,
//! animated identity. The event loop is a simple blocking poll (~30fps) so we
//! pull in no async runtime.

use crate::{
    brand,
    config::{self, LaunchConfig},
    detect,
    logo,
    prepare::{self, BuildJob, BuildState},
    requirements,
    terminal::TerminalGuard,
};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};
use std::time::Duration;

const TARGET_FPS_MS: u64 = 33;
const MIN_W: u16 = 78;
const MIN_H: u16 = 28;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Welcome,
    Form,
    Identity,
    Build,
    Ready,
    Error,
}

enum Flow {
    Continue,
    Quit,
    Launch(LaunchConfig),
}

struct Form {
    run_id: String,
    server_host: String,
    server_port: String,
    slot: String,
    micro_batch: String,
    device_idx: usize,
    /// 0..=6 (6 is the Start button).
    focus: usize,
}

pub struct App {
    screen: Screen,
    frame: u64,
    host_summary: String,
    devices: Vec<detect::DeviceOption>,
    form: Form,
    identity_path: Option<std::path::PathBuf>,
    identity_created: Option<bool>,
    identity_error: Option<String>,
    build: Option<BuildJob>,
    build_elapsed: std::time::Duration,
    build_failed_msg: String,
    launch: Option<LaunchConfig>,
    form_error: Option<String>,
}

impl App {
    pub fn new() -> Self {
        let devices = detect::detect_devices();
        // Pre-fill the micro-batch from the best GPU's VRAM when we can; fall
        // back to the conservative default otherwise. The field stays editable.
        let micro_batch = requirements::recommended_micro_batch(detect::best_gpu_vram_mib())
            .map(|n| n.to_string())
            .unwrap_or_else(|| config::DEFAULT_MICRO_BATCH.to_string());
        Self {
            screen: Screen::Welcome,
            frame: 0,
            host_summary: detect::host_summary(),
            devices,
            form: Form {
                run_id: config::DEFAULT_RUN_ID.to_string(),
                server_host: config::DEFAULT_SERVER_HOST.to_string(),
                server_port: config::DEFAULT_SERVER_PORT.to_string(),
                slot: config::DEFAULT_SLOT.to_string(),
                micro_batch,
                device_idx: 0,
                focus: 0,
            },
            identity_path: None,
            identity_created: None,
            identity_error: None,
            build: None,
            build_elapsed: std::time::Duration::ZERO,
            build_failed_msg: String::new(),
            launch: None,
            form_error: None,
        }
    }

    pub fn drive(&mut self, guard: &mut TerminalGuard) -> Result<Option<LaunchConfig>> {
        loop {
            self.frame = self.frame.wrapping_add(1);
            self.auto_advance();

            guard.term().draw(|f| self.render(f))?;

            while event::poll(Duration::from_millis(TARGET_FPS_MS))? {
                if let Event::Key(k) = event::read()? {
                    match self.handle_key(k) {
                        Flow::Continue => {}
                        Flow::Quit => return Ok(None),
                        Flow::Launch(cfg) => return Ok(Some(cfg)),
                    }
                }
            }
        }
    }

    // --- per-frame automatic transitions ------------------------------------

    fn auto_advance(&mut self) {
        if self.screen == Screen::Build {
            if let Some(job) = &self.build {
                let snap = job.snapshot();
                self.build_elapsed = snap.elapsed;
                match snap.state {
                    BuildState::Success => {
                        // Guard against a "successful" build that still
                        // can't load libtorch (shouldn't happen after a
                        // forced rebuild, but don't hand off a broken bin).
                        if prepare::client_runs() {
                            self.screen = Screen::Ready;
                        } else {
                            self.build_failed_msg = "build finished but the \
                                binary still fails to load libtorch"
                                .to_string();
                            self.screen = Screen::Error;
                        }
                    }
                    BuildState::Failed(msg) => {
                        self.build_failed_msg = msg;
                        self.screen = Screen::Error;
                    }
                    BuildState::Running => {}
                }
            }
        }
    }

    // --- input --------------------------------------------------------------

    fn handle_key(&mut self, k: KeyEvent) -> Flow {
        // Global quit.
        if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
            return Flow::Quit;
        }

        match self.screen {
            Screen::Welcome => match k.code {
                KeyCode::Enter | KeyCode::Right | KeyCode::Tab => self.screen = Screen::Form,
                KeyCode::Char('q') | KeyCode::Esc => return Flow::Quit,
                _ => {}
            },
            Screen::Form => return self.handle_form_key(k),
            Screen::Identity => match k.code {
                KeyCode::Enter | KeyCode::Tab => return self.begin_build(),
                KeyCode::Char('b') | KeyCode::Left | KeyCode::Esc => self.screen = Screen::Form,
                KeyCode::Char('q') => return Flow::Quit,
                _ => {}
            },
            Screen::Build => match k.code {
                KeyCode::Char('q') | KeyCode::Esc => return Flow::Quit,
                _ => {}
            },
            Screen::Ready => match k.code {
                KeyCode::Enter => return self.finalize_launch(),
                KeyCode::Char('q') | KeyCode::Esc => return Flow::Quit,
                _ => {}
            },
            Screen::Error => match k.code {
                KeyCode::Char('r') => return self.begin_build(),
                KeyCode::Char('q') | KeyCode::Esc => return Flow::Quit,
                _ => {}
            },
        }
        Flow::Continue
    }

    fn handle_form_key(&mut self, k: KeyEvent) -> Flow {
        let last = 6; // index of the Start button
        // Text fields: indices 0..=3 and 5. Device is 4. Start is 6.
        match k.code {
            KeyCode::Down | KeyCode::Tab => self.form.focus = (self.form.focus + 1) % (last + 1),
            KeyCode::Up => {
                self.form.focus = if self.form.focus == 0 {
                    last
                } else {
                    self.form.focus - 1
                }
            }
            KeyCode::Enter => {
                if self.form.focus == last {
                    return self.proceed_to_identity();
                } else if self.form.focus < last {
                    self.form.focus += 1;
                }
            }
            KeyCode::Esc => return Flow::Quit,
            KeyCode::Char('q') if self.form.focus == last => return Flow::Quit,
            KeyCode::Left if self.form.focus == 4 => self.cycle_device(-1),
            KeyCode::Right if self.form.focus == 4 => self.cycle_device(1),
            KeyCode::Backspace => {
                if let Some(s) = self.text_field_mut() {
                    s.pop();
                }
            }
            KeyCode::Char(c) => {
                let accepted = self.accept_char(c);
                if accepted {
                    if let Some(s) = self.text_field_mut() {
                        s.push(c);
                    }
                }
            }
            _ => {}
        }
        Flow::Continue
    }

    fn text_field_mut(&mut self) -> Option<&mut String> {
        match self.form.focus {
            0 => Some(&mut self.form.run_id),
            1 => Some(&mut self.form.server_host),
            2 => Some(&mut self.form.server_port),
            3 => Some(&mut self.form.slot),
            5 => Some(&mut self.form.micro_batch),
            _ => None,
        }
    }

    fn accept_char(&self, c: char) -> bool {
        match self.form.focus {
            2 | 5 => c.is_ascii_digit(),
            0 | 3 => c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'),
            1 => c.is_ascii_alphanumeric() || matches!(c, '-' | '.'),
            _ => false,
        }
    }

    fn cycle_device(&mut self, dir: i32) {
        let n = self.devices.len();
        if n == 0 {
            return;
        }
        let i = self.form.device_idx as i32;
        self.form.device_idx = ((i + dir).rem_euclid(n as i32)) as usize;
        // Changing the device may resolve the VRAM-gate error; clear it so the
        // message only shows while the offending device is selected.
        self.form_error = None;
    }

    fn proceed_to_identity(&mut self) -> Flow {
        // Hardware gate: a CUDA selection below the VRAM floor would OOM
        // seconds into the run, so refuse to advance and explain why.
        let dev = self.devices[self.form.device_idx].clone();
        if !requirements::meets_minimum(dev.vram_mib) {
            let mib = dev.vram_mib.unwrap_or(0);
            self.form_error = Some(format!(
                "{} ({} MiB) is below the {} MiB minimum — choose another device.",
                dev.value, mib, requirements::MIN_VRAM_MIB
            ));
            return Flow::Continue;
        }
        self.form_error = None;

        // Validate.
        if self.form.run_id.trim().is_empty() {
            return Flow::Continue;
        }
        if self.form.server_host.trim().is_empty() {
            return Flow::Continue;
        }
        if self.form.server_port.parse::<u16>().is_err() {
            return Flow::Continue;
        }
        if self.form.micro_batch.parse::<usize>().is_err() {
            return Flow::Continue;
        }

        let slot_dir = config::slot_dir(&self.form.slot);
        let identity_key = slot_dir.join("identity.key");
        match config::ensure_identity_key(&identity_key) {
            Ok(created) => {
                self.identity_path = Some(identity_key);
                self.identity_created = Some(created);
                self.identity_error = None;
                self.screen = Screen::Identity;
            }
            Err(e) => {
                self.identity_error = Some(format!("{e:#}"));
            }
        }
        Flow::Continue
    }

    fn begin_build(&mut self) -> Flow {
        // Drop any previous job (only reachable from Error, where it already
        // exited, so no concurrent builds to the same target dir).
        self.build = None;
        self.build_failed_msg.clear();
        self.build_elapsed = std::time::Duration::ZERO;

        let bin_exists = config::client_bin().exists();
        if bin_exists
            && prepare::client_runs()
            && !prepare::torch_changed_since_build()
        {
            // Binary is present and matches the active libtorch — reuse it.
            self.screen = Screen::Ready;
            return Flow::Continue;
        }
        // Either missing, fails to load, or stale relative to a torch
        // upgrade/downgrade. If a binary exists we must clean torch-sys so it
        // re-links against the current torch rather than reusing cached links.
        self.build = Some(BuildJob::start(config::CLIENT_CRATE, bin_exists));
        self.screen = Screen::Build;
        Flow::Continue
    }

    fn finalize_launch(&mut self) -> Flow {
        let port = match self.form.server_port.parse::<u16>() {
            Ok(p) => p,
            Err(_) => return Flow::Continue,
        };
        let slot_dir = config::slot_dir(&self.form.slot);
        let identity_key = self
            .identity_path
            .clone()
            .unwrap_or_else(|| slot_dir.join("identity.key"));
        let log_dir = slot_dir.join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let log_file = log_dir.join("client.log");
        let device = self
            .devices
            .get(self.form.device_idx)
            .map(|d| d.value.clone())
            .unwrap_or_else(|| "auto".to_string());
        let cfg = LaunchConfig {
            run_id: self.form.run_id.clone(),
            server_addr: format!("{}:{}", self.form.server_host, port),
            device,
            micro_batch_size: self.form.micro_batch.clone(),
            identity_key,
            log_file,
        };
        self.launch = Some(cfg.clone());
        Flow::Launch(cfg)
    }

    // --- rendering ----------------------------------------------------------

    fn render(&mut self, f: &mut Frame) {
        let area = f.area();

        if area.width < MIN_W || area.height < MIN_H {
            draw_too_small(f, area);
            return;
        }

        draw_header(f, area, self.host_summary.as_str(), self.screen);

        // Reserve header (4) at top and footer (1) at bottom; screens render in
        // the middle `inner` region.
        let inner = Layout::vertical([
            Constraint::Length(4),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(area)[1];

        match self.screen {
            Screen::Welcome => self.render_welcome(f, inner),
            Screen::Form => self.render_form(f, inner),
            Screen::Identity => self.render_identity(f, inner),
            Screen::Build => self.render_build(f, inner),
            Screen::Ready => self.render_ready(f, inner),
            Screen::Error => self.render_error(f, inner),
        }

        draw_footer(f, area, self.screen);
    }

    fn render_welcome(&self, f: &mut Frame, area: Rect) {
        let cx = area.x + area.width / 2;
        let cta = "Press Enter to configure your node";
        let logo_fits = area.width >= logo::width() && area.height >= logo::height() + 2;

        if logo_fits {
            let pitch = [
                "Donate idle compute to train open foundation models.",
                "No account. No data leaves your machine beyond model weights.",
            ];
            let host = format!("Host: {}", self.host_summary);
            let group_h = logo::height() + 1 + pitch.len() as u16 + 1 + 1;
            let content_h = area.height.saturating_sub(2); // leave bottom line for the CTA
            let show_text = content_h >= group_h;
            let logo_y = if show_text {
                area.y + content_h.saturating_sub(group_h) / 2
            } else {
                area.y + content_h.saturating_sub(logo::height()) / 2
            };

            let logo_area = Rect {
                x: area.x,
                y: logo_y,
                width: area.width,
                height: logo::height(),
            };
            logo::draw(f.buffer_mut(), logo_area, self.frame);

            if show_text {
                let text_y = logo_y + logo::height() + 1;
                for (i, line) in pitch.iter().enumerate() {
                    draw_centered_line(
                        f,
                        area,
                        text_y + i as u16,
                        line,
                        Style::default().fg(brand::INK),
                    );
                }
                draw_centered_line(
                    f,
                    area,
                    text_y + pitch.len() as u16 + 1,
                    &host,
                    Style::default().fg(brand::DIM),
                );
            }

            f.buffer_mut().set_string(
                cx.saturating_sub(str_w(cta) / 2),
                area.y + area.height.saturating_sub(1),
                cta,
                Style::default()
                    .fg(brand::INK)
                    .add_modifier(Modifier::BOLD),
            );
            return;
        }

        let pct = |h: u16| Constraint::Percentage(h);
        let col = Layout::vertical([pct(20), pct(60), pct(20)]).split(area);
        let mid = col[1];

        let cx = mid.x + mid.width / 2;

        let wordmark = "◆ AETHERCOMPUTE";
        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(wordmark) / 2),
            mid.y,
            wordmark,
            Style::default()
                .fg(brand::BRAND_A)
                .add_modifier(Modifier::BOLD),
        );

        let pitch = [
            "Donate your idle GPU to train open foundation models.",
            "Your node joins a global, decentralized training run and",
            "contributes gradients alongside hundreds of other volunteers.",
            "",
            "No account. No data leaves your machine beyond model weights.",
            "Quit any time — your spot opens for the next epoch.",
        ];
        let max_w = pitch.iter().map(|l| str_w(l)).max().unwrap_or(0);
        let block_x = cx.saturating_sub(max_w / 2);
        for (i, l) in pitch.iter().enumerate() {
            f.buffer_mut().set_string(
                block_x,
                mid.y + 3 + i as u16,
                l,
                Style::default().fg(brand::INK),
            );
        }

        let host = format!("Host: {}", self.host_summary);
        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(&host) / 2),
            mid.y + mid.height.saturating_sub(3),
            &host,
            Style::default().fg(brand::DIM),
        );

        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(cta) / 2),
            mid.y + mid.height.saturating_sub(1),
            cta,
            Style::default()
                .fg(brand::BRAND_A)
                .add_modifier(Modifier::BOLD),
        );
    }

    fn render_form(&self, f: &mut Frame, area: Rect) {
        draw_section_title(f, area, "Configure Your Node");

        let dev = &self.devices[self.form.device_idx];
        let rows: Vec<(&str, String)> = vec![
            ("Run ID", self.form.run_id.clone()),
            ("Server Host", self.form.server_host.clone()),
            ("Server Port", self.form.server_port.clone()),
            ("Client Slot", self.form.slot.clone()),
            ("Device", dev.label.clone()),
            ("Micro Batch Size", self.form.micro_batch.clone()),
        ];

        let inner = Rect {
            x: area.x + 2,
            y: area.y + 4,
            width: area.width.saturating_sub(4),
            height: area.height.saturating_sub(5),
        };
        let row_h: u16 = 3;
        for (i, (label, value)) in rows.iter().enumerate() {
            let r = Rect {
                x: inner.x,
                y: inner.y + (i as u16) * row_h,
                width: inner.width,
                height: row_h,
            };
            self.draw_field(f, r, label, value, self.form.focus == i);
        }

        // Start button.
        let btn = Rect {
            x: inner.x,
            y: inner.y + (rows.len() as u16) * row_h + 1,
            width: inner.width,
            height: 3,
        };
        let focused = self.form.focus == 6;
        let border_col = if focused {
            brand::DIM
        } else {
            brand::PANEL_HI
        };
        let label = "Start Training";
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Plain)
            .border_style(Style::default().fg(border_col));
        let p = Paragraph::new(label)
            .alignment(Alignment::Center)
            .style(Style::default().fg(if focused { brand::INK } else { brand::DIM }))
            .block(block);
        f.render_widget(p, btn);

        // Inline hardware-gate message (e.g. below-minimum VRAM GPU selected).
        if let Some(msg) = &self.form_error {
            let y = btn.y + btn.height;
            draw_centered_line(f, inner, y, msg, Style::default().fg(brand::DANGER));
        }
    }

    fn draw_field(&self, f: &mut Frame, r: Rect, label: &str, value: &str, focused: bool) {
        let border_col = if focused {
            brand::DIM
        } else {
            brand::PANEL_HI
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_col))
            .title(Span::styled(
                format!(" {label} "),
                Style::default().fg(if focused { brand::INK } else { brand::DIM }),
            ));
        let val_style = Style::default().fg(brand::INK);
        let p = Paragraph::new(value).alignment(Alignment::Left).style(val_style).block(block);
        f.render_widget(p, r);
    }

    fn render_identity(&self, f: &mut Frame, area: Rect) {
        draw_section_title(f, area, "Node Identity");
        let lines = Layout::vertical([Constraint::Min(0)]).split(area);
        let b = lines[0];
        let cx = b.x + b.width / 2;

        let banner = match self.identity_created {
            Some(true) => ("Generated a fresh identity", brand::SUCCESS),
            Some(false) => ("Using existing identity", brand::INK),
            None => ("Identity", brand::INK),
        };
        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(banner.0) / 2),
            b.y + 4,
            banner.0,
            Style::default().fg(banner.1).add_modifier(Modifier::BOLD),
        );
        if let Some(p) = &self.identity_path {
            let s = format!("Key: {}", p.display());
            f.buffer_mut().set_string(
                cx.saturating_sub(str_w(&s) / 2),
                b.y + 6,
                &s,
                Style::default().fg(brand::DIM),
            );
        }
        if let Some(e) = &self.identity_error {
            f.buffer_mut().set_string(
                b.x + 2,
                b.y + 8,
                e,
                Style::default().fg(brand::DANGER),
            );
        }

        let note = [
            "Your identity key is a 32-byte secret stored on disk. It is your",
            "membership in the run — back it up if you care about your node ID.",
            "It never leaves this machine.",
        ];
        let max_w = note.iter().map(|l| str_w(l)).max().unwrap_or(0);
        let note_x = cx.saturating_sub(max_w / 2);
        for (i, l) in note.iter().enumerate() {
            f.buffer_mut().set_string(
                note_x,
                b.y + 9 + i as u16,
                l,
                Style::default().fg(brand::DIM),
            );
        }

        let cta = "Enter: build and launch · B: back · Q: quit";
        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(cta) / 2),
            b.y + b.height.saturating_sub(2),
            cta,
            Style::default()
                .fg(brand::BRAND_A)
                .add_modifier(Modifier::BOLD),
        );
    }

    fn render_build(&mut self, f: &mut Frame, area: Rect) {
        draw_section_title(f, area, "Compile Training Engine");
        let snap = self.build.as_ref().map(|b| b.snapshot());

        let cx = area.x + area.width / 2;

        let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let s = spinner[(self.frame as usize) % spinner.len()];
        let msg = match &snap {
            Some(s) if matches!(s.state, BuildState::Success) => "Engine ready".to_string(),
            _ => format!("Compiling {} crates…", snap.as_ref().map(|s| s.compiles).unwrap_or(0)),
        };
        // Render spinner + message as a single centered unit.
        let unit_w = 3u16 + str_w(&msg); // spinner + two spaces + message
        let unit_x = cx.saturating_sub(unit_w / 2);
        f.buffer_mut().set_string(
            unit_x,
            area.y + 4,
            s,
            Style::default()
                .fg(brand::BRAND_B)
                .add_modifier(Modifier::BOLD),
        );
        f.buffer_mut().set_string(
            unit_x + 3,
            area.y + 4,
            &msg,
            Style::default().fg(brand::INK),
        );

        // Calm monochrome indeterminate sweep.
        let bar_w = area.width.saturating_sub(8);
        let bar = Rect {
            x: area.x + 4,
            y: area.y + 6,
            width: bar_w,
            height: 1,
        };
        draw_progress_bar(f, bar, self.frame);

        if let Some(snap) = &snap {
            let t = format!("Elapsed {:#?}", snap.elapsed);
            f.buffer_mut().set_string(
                area.x + 4,
                area.y + 8,
                &t,
                Style::default().fg(brand::DIM),
            );

            let log_h = area.height.saturating_sub(13).max(4);
            let log = Rect {
                x: area.x + 4,
                y: area.y + 10,
                width: area.width.saturating_sub(8),
                height: log_h,
            };
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(brand::PANEL_HI))
                .title(Span::styled(
                    " Build Log ",
                    Style::default().fg(brand::DIM),
                ))
                .style(Style::default().bg(brand::PANEL));
            let lines: Vec<Line> = snap
                .lines
                .iter()
                .map(|l| {
                    let col = if l.contains("error") || l.contains("warning") {
                        brand::WARN
                    } else {
                        brand::DIM
                    };
                    Line::from(Span::styled(l.clone(), Style::default().fg(col)))
                })
                .collect();
            let p = Paragraph::new(lines).block(block);
            f.render_widget(p, log);
        }
    }

    fn render_ready(&self, f: &mut Frame, area: Rect) {
        draw_section_title(f, area, "Ready to Train");
        let lines = Layout::vertical([Constraint::Min(0)]).split(area);
        let b = lines[0];
        let dev = &self.devices[self.form.device_idx];
        let mut rows = vec![
            ("Run ID", self.form.run_id.clone()),
            ("Server", format!("{}:{}", self.form.server_host, self.form.server_port)),
            ("Device", dev.value.clone()),
            ("Micro Batch Size", self.form.micro_batch.clone()),
        ];
        if let Some(p) = &self.identity_path {
            rows.push(("Identity Key", p.display().to_string()));
        }
        rows.push((
            "Engine",
            config::client_bin().display().to_string(),
        ));

        let start_y = b.y + 4;
        for (i, (k, v)) in rows.iter().enumerate() {
            let y = start_y + i as u16;
            f.buffer_mut().set_string(
                b.x + 6,
                y,
                k,
                Style::default().fg(brand::DIM),
            );
            f.buffer_mut().set_string(
                b.x + 22,
                y,
                v,
                Style::default().fg(brand::INK),
            );
        }

        let cx = b.x + b.width / 2;
        let cta = "Press Enter to launch the client";
        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(cta) / 2),
            b.y + b.height.saturating_sub(3),
            cta,
            Style::default()
                .fg(brand::BRAND_A)
                .add_modifier(Modifier::BOLD),
        );
        let note = "The terminal will switch to the training dashboard.";
        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(note) / 2),
            b.y + b.height.saturating_sub(1),
            note,
            Style::default().fg(brand::DIM),
        );
    }

    fn render_error(&self, f: &mut Frame, area: Rect) {
        draw_section_title(f, area, "Build Failed");
        let lines = Layout::vertical([Constraint::Min(0)]).split(area);
        let b = lines[0];
        f.buffer_mut().set_string(
            b.x + 2,
            b.y + 4,
            &self.build_failed_msg,
            Style::default().fg(brand::DANGER).add_modifier(Modifier::BOLD),
        );

        let snap_lines = self
            .build
            .as_ref()
            .map(|j| j.snapshot().lines)
            .unwrap_or_default();
        let log_h = b.height.saturating_sub(9).max(4);
        let log = Rect {
            x: b.x + 2,
            y: b.y + 6,
            width: b.width.saturating_sub(4),
            height: log_h,
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(brand::PANEL_HI))
            .title(Span::styled(" Recent output ", Style::default().fg(brand::DIM)))
            .style(Style::default().bg(brand::PANEL));
        let lp: Vec<Line> = snap_lines
            .iter()
            .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(brand::INK))))
            .collect();
        f.render_widget(Paragraph::new(lp).block(block), log);

        let cx = b.x + b.width / 2;
        let cta = "R: retry · Q: quit";
        f.buffer_mut().set_string(
            cx.saturating_sub(str_w(cta) / 2),
            b.y + b.height.saturating_sub(2),
            cta,
            Style::default()
                .fg(brand::WARN)
                .add_modifier(Modifier::BOLD),
        );
    }
}

// --- free-standing draw helpers -------------------------------------------

pub fn run() -> Result<Option<LaunchConfig>> {
    let mut guard = TerminalGuard::init()?;
    let mut app = App::new();
    let result = app.drive(&mut guard);
    guard.restore();
    if let Err(e) = &result {
        eprintln!("aether-volunteer: {e:#}");
    }
    result
}

fn draw_header(f: &mut Frame, area: Rect, host: &str, screen: Screen) {
    let h = Layout::vertical([Constraint::Length(4), Constraint::Min(0)]).split(area)[0];
    let wordmark = "◆ AETHERCOMPUTE";
    f.buffer_mut().set_string(
        h.x + 3,
        1,
        wordmark,
        Style::default()
            .fg(brand::BRAND_A)
            .add_modifier(Modifier::BOLD),
    );
    let after_wm = h.x + 3 + str_w(wordmark) + 2;
    f.buffer_mut().set_string(
        after_wm,
        1,
        host,
        Style::default().fg(brand::DIM),
    );
    let step = match screen {
        Screen::Welcome => "1/5",
        Screen::Form => "2/5",
        Screen::Identity => "3/5",
        Screen::Build => "4/5",
        Screen::Ready | Screen::Error => "5/5",
    };
    let s = format!("Step {step} of 5");
    f.buffer_mut().set_string(
        h.x + h.width.saturating_sub(str_w(&s) + 2),
        1,
        &s,
        Style::default().fg(brand::DIM),
    );
    // Header divider.
    for x in h.x..h.x + h.width {
        let cell = &mut f.buffer_mut()[(x, 3)];
        cell.set_char('─')
            .set_style(Style::default().fg(brand::PANEL_HI));
    }
}

fn draw_footer(f: &mut Frame, area: Rect, screen: Screen) {
    let y = area.y + area.height.saturating_sub(1);
    let hint = match screen {
        Screen::Welcome => "Enter: continue · Q: quit",
        Screen::Form => "Up/Down: navigate · Left/Right: device · Enter: next · Q: quit",
        Screen::Identity => "Enter: build and launch · B: back · Q: quit",
        Screen::Build => "Compiling… · Q: abort",
        Screen::Ready => "Enter: launch · Q: quit",
        Screen::Error => "R: retry · Q: quit",
    };
    f.buffer_mut().set_string(
        area.x + 2,
        y,
        hint,
        Style::default().fg(brand::DIM),
    );
    let url = "aethercompute.org";
    f.buffer_mut().set_string(
        area.x + area.width.saturating_sub(url.len() as u16 + 2),
        y,
        url,
        Style::default().fg(brand::BRAND_B),
    );
}

fn draw_section_title(f: &mut Frame, area: Rect, title: &str) {
    let x = area.x + 2;
    let y = area.y + 1;
    f.buffer_mut().set_string(
        x,
        y,
        title,
        Style::default()
            .fg(brand::INK)
            .add_modifier(Modifier::BOLD),
    );
    let underline_w = str_w(title).max(10);
    let max_w = area.width.saturating_sub(4);
    for dx in 0..underline_w.min(max_w) {
        let cell = &mut f.buffer_mut()[(x + dx, y + 1)];
        cell.set_char('─')
            .set_style(Style::default().fg(brand::PANEL_HI));
    }
}

fn draw_centered_line(f: &mut Frame, area: Rect, y: u16, text: &str, style: Style) {
    if y >= area.y + area.height {
        return;
    }
    let x = area.x + area.width.saturating_sub(str_w(text)) / 2;
    f.buffer_mut().set_string(x, y, text, style);
}

fn draw_progress_bar(f: &mut Frame, area: Rect, frame: u64) {
    let w = area.width as usize;
    if w == 0 {
        return;
    }
    let tail = 10usize.min(w);
    let span = w + tail;
    let head = ((frame as f32 * 0.6) as usize) % span;
    for x in 0..w {
        let d = head as i32 - x as i32;
        let (ch, col) = if d >= 0 && d < tail as i32 {
            let fade = 1.0 - d as f32 / tail as f32;
            (
                '█',
                brand::lerp_color(brand::PANEL_HI, brand::BRAND_A, fade),
            )
        } else {
            ('░', brand::PANEL_HI)
        };
        f.buffer_mut()
            .set_string(area.x + x as u16, area.y, ch.to_string(), Style::default().fg(col));
    }
}

fn draw_too_small(f: &mut Frame, area: Rect) {
    let msg = format!(
        "please resize your terminal (need ≥{}x{}, have {}x{})",
        MIN_W,
        MIN_H,
        area.width,
        area.height
    );
    let x = area.x + area.width.saturating_sub(str_w(&msg)) / 2;
    let y = area.y + area.height / 2;
    f.buffer_mut()
        .set_string(x, y, &msg, Style::default().fg(brand::WARN));
}

/// Character width of a string — used for centering. (Most glyphs used here are
/// single-cell, so char count is a close enough proxy for display width and,
/// unlike `str::len`, it is not confused by multi-byte UTF-8 such as `►`, `·`
/// or `—`.)
fn str_w(s: &str) -> u16 {
    s.chars().count() as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    /// Every screen must render to a fixed-size buffer without panicking
    /// (catches out-of-bounds buffer writes, bad rect math, unwrap-on-None).
    #[test]
    fn renders_every_screen_without_panic() {
        let mut app = App::new();
        let mut term = Terminal::new(TestBackend::new(110, 32)).unwrap();
        let screens = [
            Screen::Welcome,
            Screen::Form,
            Screen::Identity,
            Screen::Build,
            Screen::Ready,
            Screen::Error,
        ];
        for screen in screens {
            app.screen = screen;
            app.frame = 12;
            term.draw(|f| app.render(f)).unwrap();
        }
    }

    /// The too-small guard must render instead of crashing on a tiny terminal.
    #[test]
    fn renders_too_small_guard() {
        let mut app = App::new();
        let mut term = Terminal::new(TestBackend::new(40, 8)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
    }

    #[test]
    fn form_validation_rejects_bad_port() {
        let mut app = App::new();
        app.screen = Screen::Form;
        app.form.server_port = "notaport".to_string();
        app.form.focus = 6; // Start button
        let flow = app.handle_key(KeyEvent::new(
            event::KeyCode::Enter,
            event::KeyModifiers::NONE,
        ));
        assert!(matches!(flow, Flow::Continue));
        assert_eq!(app.screen, Screen::Form); // did not advance
    }

    #[test]
    fn below_min_gpu_blocks_start_and_clears_on_change() {
        let mut app = App::new();
        app.screen = Screen::Form;
        // Inject a deterministic device list (App::new reads the real host).
        app.devices = vec![
            detect::DeviceOption { value: "auto".into(), label: "Auto".into(), tag: "AUTO", vram_mib: None },
            detect::DeviceOption { value: "cuda:0".into(), label: "Weak GPU".into(), tag: "CUDA", vram_mib: Some(2048) },
            detect::DeviceOption { value: "cuda:1".into(), label: "Big GPU".into(), tag: "CUDA", vram_mib: Some(24 * 1024) },
        ];
        // Select the below-minimum GPU and press Start (focus 6 = Start button).
        app.form.device_idx = 1;
        app.form.focus = 6;
        let flow = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(flow, Flow::Continue));
        assert_eq!(app.screen, Screen::Form); // gated — did not advance
        assert!(app.form_error.is_some());

        // Moving to a qualifying GPU clears the error.
        app.cycle_device(1);
        assert_eq!(app.form.device_idx, 2);
        assert!(app.form_error.is_none());
    }
}
