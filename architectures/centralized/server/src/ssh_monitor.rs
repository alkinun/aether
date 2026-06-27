use anyhow::Result;
use psyche_coordinator::{
    model::{Checkpoint, LLMArchitecture},
    ClientState, RunState,
};
use russh::keys::ssh_key::{Algorithm, PublicKey};
use russh::server::*;
use russh::{Channel, ChannelId, Pty};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::web::{LossPoint, SharedState, WebState};

const BRAND_A: &str = "\x1b[38;2;218;78;138m";
const BRAND_B: &str = "\x1b[38;2;82;184;205m";
const ACCENT_AMBER: &str = "\x1b[38;2;226;136;68m";
const BLOOM_BONE: &str = "\x1b[38;2;226;204;184m";
const DIM: &str = "\x1b[38;2;116;98;104m";
const PANEL_HI: &str = "\x1b[38;2;70;56;64m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

struct ClientView {
    handle: Handle,
    channel: ChannelId,
    width: u16,
    height: u16,
    frame: u64,
}

#[derive(Clone)]
struct MonitorServer {
    state: SharedState,
    clients: Arc<Mutex<HashMap<usize, ClientView>>>,
    id: usize,
}

impl MonitorServer {
    fn new(state: SharedState) -> Self {
        Self {
            state,
            clients: Arc::new(Mutex::new(HashMap::new())),
            id: 0,
        }
    }

    async fn run(
        &mut self,
        port: u16,
        host_key_path: Option<PathBuf>,
        cancel: CancellationToken,
    ) -> Result<()> {
        self.start_render_loop(cancel.clone());

        let config = Config {
            inactivity_timeout: Some(Duration::from_secs(3600)),
            auth_rejection_time: Duration::from_millis(250),
            auth_rejection_time_initial: Some(Duration::from_millis(0)),
            keys: vec![load_or_generate_host_key(host_key_path)?],
            nodelay: true,
            ..Default::default()
        };

        tokio::select! {
            result = self.run_on_address(Arc::new(config), ("0.0.0.0", port)) => result.map_err(Into::into),
            _ = cancel.cancelled() => Ok(()),
        }
    }

    fn start_render_loop(&self, cancel: CancellationToken) {
        let clients = self.clients.clone();
        let state = self.state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(1000));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = ticker.tick() => {
                        let snapshot = snapshot_state(&state);
                        let mut clients = clients.lock().await;
                        for client in clients.values_mut() {
                            client.frame = client.frame.wrapping_add(1);
                            let data = render_ansi(&snapshot, client.width, client.height, client.frame);
                            if let Err(err) = client.handle.data(client.channel, data.into_bytes().into()).await {
                                warn!(?err, "failed to send ssh monitor frame");
                            }
                        }
                    }
                }
            }
        });
    }

    async fn draw_client(&mut self) {
        let snapshot = snapshot_state(&self.state);
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get_mut(&self.id) {
            let data = render_ansi(&snapshot, client.width, client.height, client.frame);
            if let Err(err) = client
                .handle
                .data(client.channel, data.into_bytes().into())
                .await
            {
                warn!(?err, "failed to send ssh monitor frame");
            }
        }
    }

    async fn resize(&mut self, col_width: u32, row_height: u32) {
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get_mut(&self.id) {
            client.width = col_width.max(1).min(u16::MAX as u32) as u16;
            client.height = row_height.max(1).min(u16::MAX as u32) as u16;
        }
    }
}

impl Server for MonitorServer {
    type Handler = Self;

    fn new_client(&mut self, _: Option<std::net::SocketAddr>) -> Self {
        let mut handler = self.clone();
        handler.id = self.id;
        self.id += 1;
        handler
    }
}

impl Handler for MonitorServer {
    type Error = anyhow::Error;

    async fn channel_open_session(
        &mut self,
        channel: Channel<Msg>,
        session: &mut Session,
    ) -> Result<bool, Self::Error> {
        self.clients.lock().await.insert(
            self.id,
            ClientView {
                handle: session.handle(),
                channel: channel.id(),
                width: 80,
                height: 24,
                frame: 0,
            },
        );
        Ok(true)
    }

    async fn auth_none(&mut self, _: &str) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn auth_password(&mut self, _: &str, _: &str) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn auth_publickey_offered(
        &mut self,
        _: &str,
        _: &PublicKey,
    ) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn auth_publickey(&mut self, _: &str, _: &PublicKey) -> Result<Auth, Self::Error> {
        Ok(Auth::Accept)
    }

    async fn shell_request(
        &mut self,
        channel: ChannelId,
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        session.channel_success(channel)?;
        self.draw_client().await;
        Ok(())
    }

    async fn exec_request(
        &mut self,
        channel: ChannelId,
        _: &[u8],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        session.channel_success(channel)?;
        self.draw_client().await;
        Ok(())
    }

    async fn env_request(
        &mut self,
        channel: ChannelId,
        _: &str,
        _: &str,
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        session.channel_success(channel)?;
        Ok(())
    }

    async fn data(
        &mut self,
        channel: ChannelId,
        data: &[u8],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        if data == b"q" || data == [3] {
            self.clients.lock().await.remove(&self.id);
            session.close(channel)?;
        }
        Ok(())
    }

    async fn channel_close(&mut self, _: ChannelId, _: &mut Session) -> Result<(), Self::Error> {
        self.clients.lock().await.remove(&self.id);
        Ok(())
    }

    async fn window_change_request(
        &mut self,
        _: ChannelId,
        col_width: u32,
        row_height: u32,
        _: u32,
        _: u32,
        _: &mut Session,
    ) -> Result<(), Self::Error> {
        self.resize(col_width, row_height).await;
        self.draw_client().await;
        Ok(())
    }

    async fn pty_request(
        &mut self,
        channel: ChannelId,
        _: &str,
        col_width: u32,
        row_height: u32,
        _: u32,
        _: u32,
        _: &[(Pty, u32)],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        self.resize(col_width, row_height).await;
        session.channel_success(channel)?;
        self.draw_client().await;
        Ok(())
    }
}

impl Drop for MonitorServer {
    fn drop(&mut self) {
        let id = self.id;
        let clients = self.clients.clone();
        tokio::spawn(async move {
            clients.lock().await.remove(&id);
        });
    }
}

pub fn start(
    state: SharedState,
    port: u16,
    host_key_path: Option<PathBuf>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let mut server = MonitorServer::new(state);
        info!(port, "starting ssh training monitor");
        if let Err(err) = server.run(port, host_key_path, cancel).await {
            warn!(?err, "ssh training monitor stopped");
        }
    });
}

fn load_or_generate_host_key(path: Option<PathBuf>) -> Result<russh::keys::PrivateKey> {
    let Some(path) = path else {
        warn!("ssh monitor host key path not set; using ephemeral host key");
        return Ok(russh::keys::PrivateKey::random(
            &mut rand08::thread_rng(),
            Algorithm::Ed25519,
        )?);
    };

    if path.exists() {
        info!(path = %path.display(), "loading ssh monitor host key");
        return Ok(russh::keys::PrivateKey::read_openssh_file(&path)?);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let key = russh::keys::PrivateKey::random(&mut rand08::thread_rng(), Algorithm::Ed25519)?;
    key.write_openssh_file(&path, russh::keys::ssh_key::LineEnding::LF)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }

    info!(path = %path.display(), "generated ssh monitor host key");
    Ok(key)
}

struct Snapshot {
    waiting: bool,
    run_state: String,
    run_id: String,
    model: String,
    checkpoint: String,
    epoch: u32,
    step: u32,
    total_steps: u32,
    height: u32,
    epoch_steps: String,
    ready_clients: usize,
    syncing_clients: usize,
    healthy: u32,
    dropped: u32,
    withdrawn: u32,
    ejected: u32,
    server_addr: String,
    latest_loss: Option<f32>,
    latest_tps: Option<f32>,
    avg_tps: Option<f32>,
    weighted_tps: Option<f32>,
    elapsed: Option<u64>,
    remaining_steps: u32,
    tokens_per_step: Option<f64>,
    eta_weighted: Option<u64>,
    eta_current: Option<u64>,
    warmup_time: u64,
    epoch_time: u64,
    min_clients: u32,
    init_min_clients: u32,
    witness_nodes: u32,
    loss_points: Vec<(u32, f32)>,
    throughput_points: Vec<(u32, f32)>,
}

fn snapshot_state(state: &SharedState) -> Snapshot {
    let s = state.lock().unwrap();
    snapshot_from_web_state(&s)
}

fn snapshot_from_web_state(s: &WebState) -> Snapshot {
    let Some(coord) = &s.coordinator else {
        return Snapshot {
            waiting: true,
            run_state: "Waiting for coordinator data".into(),
            run_id: "-".into(),
            model: "-".into(),
            checkpoint: "-".into(),
            epoch: 0,
            step: 0,
            total_steps: 0,
            height: 0,
            epoch_steps: "-".into(),
            ready_clients: s.ready_clients.len(),
            syncing_clients: s.syncing_clients.len(),
            healthy: 0,
            dropped: 0,
            withdrawn: 0,
            ejected: 0,
            server_addr: s.server_addr.clone(),
            latest_loss: None,
            latest_tps: None,
            avg_tps: None,
            weighted_tps: None,
            elapsed: None,
            remaining_steps: 0,
            tokens_per_step: None,
            eta_weighted: None,
            eta_current: None,
            warmup_time: 0,
            epoch_time: 0,
            min_clients: 0,
            init_min_clients: 0,
            witness_nodes: 0,
            loss_points: Vec::new(),
            throughput_points: Vec::new(),
        };
    };

    let mut healthy = 0;
    let mut dropped = 0;
    let mut withdrawn = 0;
    let mut ejected = 0;
    for client in coord.epoch_state.clients.iter() {
        match client.state {
            ClientState::Healthy => healthy += 1,
            ClientState::Dropped => dropped += 1,
            ClientState::Withdrawn => withdrawn += 1,
            ClientState::Ejected => ejected += 1,
        }
    }

    let valid_tps: Vec<f32> = s
        .loss_history
        .iter()
        .filter_map(|p| p.tokens_per_sec.is_finite().then_some(p.tokens_per_sec))
        .filter(|v| *v > 0.0)
        .collect();
    let avg_tps = (!valid_tps.is_empty())
        .then(|| valid_tps.iter().copied().sum::<f32>() / valid_tps.len() as f32);
    let weighted_tps = weighted_tokens_per_sec(&s.loss_history);
    let latest = s.loss_history.last();
    let current_tps = latest.and_then(|p| p.tokens_per_sec.is_finite().then_some(p.tokens_per_sec));
    let now = current_unix_timestamp();
    let elapsed = s
        .loss_history
        .first()
        .map(|p| now.saturating_sub(p.unix_timestamp))
        .or_else(|| {
            (coord.run_state_start_unix_timestamp > 0)
                .then(|| now.saturating_sub(coord.run_state_start_unix_timestamp))
        });
    let remaining_steps = coord.config.total_steps.saturating_sub(coord.progress.step);
    let tokens_per_step = coord.get_target_global_batch_size(coord.current_round()) as f64
        * coord.get_sequence_length() as f64;
    let remaining_tokens = remaining_steps as f64 * tokens_per_step;
    let eta_weighted =
        estimate_remaining_time(remaining_tokens, weighted_tps.unwrap_or(0.0) as f64);
    let eta_current = estimate_remaining_time(remaining_tokens, current_tps.unwrap_or(0.0) as f64);
    let model = match &coord.model {
        psyche_coordinator::model::Model::LLM(llm) => format!(
            "{} · seq {}",
            format_llm_architecture(&llm.architecture),
            llm.max_seq_len
        ),
    };
    let checkpoint = match &coord.model {
        psyche_coordinator::model::Model::LLM(llm) => {
            format_checkpoint_label(&llm.checkpoint).into()
        }
    };

    Snapshot {
        waiting: false,
        run_state: format_run_state(coord.run_state).into(),
        run_id: coord.run_id.to_string(),
        model,
        checkpoint,
        epoch: coord.progress.epoch as u32,
        step: coord.progress.step,
        total_steps: coord.config.total_steps,
        height: coord.epoch_state.rounds[coord.epoch_state.rounds_head as usize].height,
        epoch_steps: format!(
            "{} - {}",
            coord.epoch_state.start_step, coord.epoch_state.last_step
        ),
        ready_clients: s.ready_clients.len(),
        syncing_clients: s.syncing_clients.len(),
        healthy,
        dropped,
        withdrawn,
        ejected,
        server_addr: s.server_addr.clone(),
        latest_loss: latest.and_then(|p| p.loss.is_finite().then_some(p.loss)),
        latest_tps: current_tps,
        avg_tps,
        weighted_tps,
        elapsed,
        remaining_steps,
        tokens_per_step: Some(tokens_per_step),
        eta_weighted,
        eta_current,
        warmup_time: coord.config.warmup_time,
        epoch_time: coord.config.epoch_time,
        min_clients: coord.config.min_clients as u32,
        init_min_clients: coord.config.init_min_clients as u32,
        witness_nodes: coord.config.witness_nodes as u32,
        loss_points: s
            .loss_history
            .iter()
            .filter_map(|p| p.loss.is_finite().then_some((p.step, p.loss)))
            .collect(),
        throughput_points: s
            .loss_history
            .iter()
            .filter_map(|p| {
                (p.tokens_per_sec.is_finite() && p.tokens_per_sec > 0.0)
                    .then_some((p.step, p.tokens_per_sec))
            })
            .collect(),
    }
}

fn render_ansi(s: &Snapshot, width: u16, height: u16, frame: u64) -> String {
    if width < 50 || height < 18 {
        return format!(
            "\x1b[2J\x1b[H{BRAND_A}{BOLD}◆ AETHERCOMPUTE{RESET}\r\n\r\n{BLOOM_BONE}Resize terminal to at least 50x18.{RESET}\r\n"
        );
    }

    let mut out = String::new();
    out.push_str("\x1b[?25l\x1b[2J\x1b[H");
    let brand = if frame % 8 == 0 { BRAND_B } else { BRAND_A };
    let state_color = if s.waiting { ACCENT_AMBER } else { BRAND_B };
    let status = if s.waiting {
        "initializing"
    } else {
        &s.run_state
    };
    line(&mut out, &format!("{brand}{BOLD}◆ AETHERCOMPUTE{RESET}  {DIM}Training Monitor{RESET}  {state_color}{status}{RESET}"));
    line(
        &mut out,
        &format!("{PANEL_HI}{}{RESET}", "─".repeat(width as usize)),
    );

    let progress = if s.total_steps > 0 {
        s.step as f32 / s.total_steps as f32
    } else {
        0.0
    };

    let overview = vec![
        kv_line("Run", &shorten(&s.run_id, 28)),
        kv_line("State", &s.run_state),
        kv_line("Epoch", &s.epoch.to_string()),
        kv_line("Step", &format!("{} / {}", s.step, s.total_steps)),
        kv_line("Height", &s.height.to_string()),
        kv_line("Epoch Steps", &s.epoch_steps),
        kv_line(
            "Server",
            if s.server_addr.is_empty() {
                "-"
            } else {
                &s.server_addr
            },
        ),
    ];
    let timing = vec![
        kv_line("Progress", &format!("{:>5.1}%", progress * 100.0)),
        color_line(
            BRAND_A,
            &progress_bar(progress, panel_value_width(width) as usize),
        ),
        kv_line("Elapsed", &format_duration_opt(s.elapsed)),
        kv_line("Remaining", &s.remaining_steps.to_string()),
        kv_line(
            "Tokens/Step",
            &s.tokens_per_step
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "-".into()),
        ),
        kv_line("ETA Weighted", &format_duration_opt(s.eta_weighted)),
        kv_line("ETA Current", &format_duration_opt(s.eta_current)),
    ];
    render_panel_pair(&mut out, width, "Overview", overview, "Timing", timing);
    blank(&mut out);

    let training = vec![
        kv_line("Latest Loss", &format_opt(s.latest_loss, 4)),
        kv_line("Tokens/sec", &format_opt(s.latest_tps, 1)),
        kv_line("Weighted tok/s", &format_opt(s.weighted_tps, 1)),
        kv_line("Average tok/s", &format_opt(s.avg_tps, 1)),
        kv_line(
            "Loss",
            &sparkline(&s.loss_points, panel_value_width(width) as usize),
        ),
        kv_line(
            "Throughput",
            &sparkline(&s.throughput_points, panel_value_width(width) as usize),
        ),
    ];
    let clients = vec![
        kv_line("Healthy", &s.healthy.to_string()),
        kv_line("Ready", &s.ready_clients.to_string()),
        kv_line("Syncing", &s.syncing_clients.to_string()),
        kv_line("Dropped", &s.dropped.to_string()),
        kv_line("Withdrawn", &s.withdrawn.to_string()),
        kv_line("Ejected", &s.ejected.to_string()),
        kv_line(
            "Total",
            &(s.healthy + s.dropped + s.withdrawn + s.ejected).to_string(),
        ),
    ];
    render_panel_pair(&mut out, width, "Training", training, "Clients", clients);
    blank(&mut out);

    if height >= 32 {
        let model = vec![
            kv_line("Model", &s.model),
            kv_line("Checkpoint", &s.checkpoint),
            kv_line("Warmup", &format_duration(s.warmup_time)),
            kv_line("Epoch Time", &format_duration(s.epoch_time)),
            kv_line("Init Min", &s.init_min_clients.to_string()),
            kv_line("Min Clients", &s.min_clients.to_string()),
            kv_line("Witnesses", &s.witness_nodes.to_string()),
        ];
        render_panel(&mut out, "Config / Model", model);
        blank(&mut out);
    }

    line(
        &mut out,
        &format!("{DIM}live SSH view · q/Ctrl-C: close · monitor.aethercompute.org{RESET}"),
    );
    out
}

fn render_panel(out: &mut String, title: &str, rows: Vec<String>) {
    let width = rows
        .iter()
        .map(|r| visible_len(r))
        .max()
        .unwrap_or(0)
        .max(title.len() + 2)
        + 4;
    let border_w = width.saturating_sub(2);
    line(
        out,
        &format!(
            "{PANEL_HI}┌─ {BLOOM_BONE}{BOLD}{title}{RESET}{PANEL_HI} {}┐{RESET}",
            "─".repeat(border_w.saturating_sub(title.len() + 3))
        ),
    );
    for row in rows {
        line(out, &boxed_line(&row, width));
    }
    line(out, &format!("{PANEL_HI}└{}┘{RESET}", "─".repeat(border_w)));
}

fn render_panel_pair(
    out: &mut String,
    width: u16,
    left_title: &str,
    mut left: Vec<String>,
    right_title: &str,
    mut right: Vec<String>,
) {
    if width < 100 {
        render_panel(out, left_title, left);
        blank(out);
        render_panel(out, right_title, right);
        return;
    }

    let col_w = (width as usize - 3) / 2;
    let left_top = panel_top(left_title, col_w);
    let right_top = panel_top(right_title, col_w);
    line(out, &format!("{left_top} {right_top}"));

    let rows = left.len().max(right.len());
    left.resize(rows, String::new());
    right.resize(rows, String::new());
    for i in 0..rows {
        line(
            out,
            &format!(
                "{} {}",
                boxed_line(&left[i], col_w),
                boxed_line(&right[i], col_w),
            ),
        );
    }
    line(
        out,
        &format!("{} {}", panel_bottom(col_w), panel_bottom(col_w)),
    );
}

fn panel_top(title: &str, width: usize) -> String {
    let inner = width.saturating_sub(2);
    let title = format!("─ {title} ");
    let fill = inner.saturating_sub(visible_len(&title));
    format!("{PANEL_HI}┌{title}{fill}┐{RESET}", fill = "─".repeat(fill))
}

fn panel_bottom(width: usize) -> String {
    format!("{PANEL_HI}└{}┘{RESET}", "─".repeat(width.saturating_sub(2)))
}

fn boxed_line(content: &str, width: usize) -> String {
    let inner = width.saturating_sub(4);
    let content = truncate_visible(content, inner);
    let pad = inner.saturating_sub(visible_len(&content));
    format!(
        "{PANEL_HI}│{RESET} {}{} {PANEL_HI}│{RESET}",
        content,
        " ".repeat(pad)
    )
}

fn kv_line(key: &str, value: &str) -> String {
    format!("{DIM}{key:<15}{RESET}{BLOOM_BONE}{value}{RESET}")
}

fn color_line(color: &str, value: &str) -> String {
    format!("{color}{value}{RESET}")
}

fn visible_len(s: &str) -> usize {
    let mut len = 0;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            len += 1;
        }
    }
    len
}

fn truncate_visible(s: &str, width: usize) -> String {
    if visible_len(s) <= width {
        return s.to_string();
    }

    let target = width.saturating_sub(1);
    let mut out = String::new();
    let mut len = 0;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            out.push(ch);
            out.push(chars.next().unwrap());
            for c in chars.by_ref() {
                out.push(c);
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else if len < target {
            out.push(ch);
            len += 1;
        } else {
            break;
        }
    }
    out.push('…');
    out.push_str(RESET);
    out
}

fn line(out: &mut String, value: &str) {
    out.push_str(value);
    out.push_str("\r\n");
}

fn blank(out: &mut String) {
    out.push_str("\r\n");
}

fn format_opt(value: Option<f32>, precision: usize) -> String {
    value
        .map(|v| format!("{v:.precision$}"))
        .unwrap_or_else(|| "-".into())
}

fn panel_value_width(width: u16) -> u16 {
    if width >= 100 {
        ((width as usize - 3) / 2).saturating_sub(23) as u16
    } else {
        width.saturating_sub(27)
    }
}

fn sparkline(points: &[(u32, f32)], width: usize) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if points.len() < 2 || width == 0 {
        return "-".into();
    }
    let width = width.max(8).min(80);
    let start = points.len().saturating_sub(width);
    let vals: Vec<f32> = points[start..].iter().map(|(_, v)| *v).collect();
    let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(f32::EPSILON);
    vals.iter()
        .map(|v| {
            let idx = (((*v - min) / range) * (BARS.len() - 1) as f32).round() as usize;
            BARS[idx.min(BARS.len() - 1)]
        })
        .collect()
}

fn current_unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn estimate_remaining_time(remaining_tokens: f64, tokens_per_sec: f64) -> Option<u64> {
    if remaining_tokens <= 0.0 {
        Some(0)
    } else if tokens_per_sec.is_finite() && tokens_per_sec > 0.0 {
        Some((remaining_tokens / tokens_per_sec).ceil() as u64)
    } else {
        None
    }
}

fn weighted_tokens_per_sec(points: &[LossPoint]) -> Option<f32> {
    let points: Vec<&LossPoint> = points
        .iter()
        .filter(|p| p.tokens_per_sec.is_finite() && p.tokens_per_sec > 0.0)
        .collect();
    if points.is_empty() {
        return None;
    }
    let mut weighted_total = 0.0;
    let mut total_weight = 0.0;
    for i in 0..points.len() {
        let weight = if i + 1 < points.len() {
            points[i + 1]
                .unix_timestamp
                .saturating_sub(points[i].unix_timestamp)
                .max(1) as f64
        } else if i > 0 {
            points[i]
                .unix_timestamp
                .saturating_sub(points[i - 1].unix_timestamp)
                .max(1) as f64
        } else {
            1.0
        };
        weighted_total += points[i].tokens_per_sec as f64 * weight;
        total_weight += weight;
    }
    (total_weight > 0.0).then(|| (weighted_total / total_weight) as f32)
}

fn format_duration_opt(secs: Option<u64>) -> String {
    secs.map(format_duration).unwrap_or_else(|| "-".into())
}

fn format_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{}h {:02}m {:02}s", h, m, s)
    } else if m > 0 {
        format!("{}m {:02}s", m, s)
    } else {
        format!("{}s", s)
    }
}

fn shorten(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.into()
    } else {
        format!(
            "{}…",
            value
                .chars()
                .take(max.saturating_sub(1))
                .collect::<String>()
        )
    }
}

fn progress_bar(progress: f32, width: usize) -> String {
    let width = width.max(8);
    let filled = ((progress.clamp(0.0, 1.0) * width as f32).round() as usize).min(width);
    format!("{}{}", "━".repeat(filled), "─".repeat(width - filled))
}

fn format_run_state(state: RunState) -> &'static str {
    match state {
        RunState::Uninitialized => "Uninitialized",
        RunState::WaitingForMembers => "Waiting for members",
        RunState::Warmup => "Warmup",
        RunState::RoundTrain => "Training",
        RunState::RoundWitness => "Witness",
        RunState::Cooldown => "Cooldown",
        RunState::Finished => "Finished",
        RunState::Paused => "Paused",
    }
}

fn format_llm_architecture(arch: &LLMArchitecture) -> &'static str {
    match arch {
        LLMArchitecture::HfLlama => "HuggingFace LLaMA",
        LLMArchitecture::HfDeepseek => "HuggingFace DeepSeek",
        LLMArchitecture::HfAuto => "HuggingFace Auto",
        LLMArchitecture::Torchtitan => "Torchtitan",
    }
}

fn format_checkpoint_label(cp: &Checkpoint) -> &'static str {
    match cp {
        Checkpoint::Ephemeral => "Ephemeral",
        Checkpoint::Dummy(_) => "Dummy",
        Checkpoint::Hub(_) => "Hub",
        Checkpoint::P2P(_) => "P2P",
        Checkpoint::Gcs(_) => "GCS",
        Checkpoint::P2PGcs(_) => "P2P+GCS",
    }
}
