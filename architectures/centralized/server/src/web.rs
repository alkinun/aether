use axum::{extract::State, response::Html, routing::get, Router};
use psyche_coordinator::{
    model::{Checkpoint, LLMArchitecture, LLMTrainingDataType, Model},
    ClientState, Coordinator, RunState, NUM_STORED_ROUNDS,
};
use psyche_core::{LearningRateSchedule, OptimizerDefinition};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize)]
pub struct LossPoint {
    pub step: u32,
    pub tokens_processed: u64,
    pub loss: f32,
    pub tokens_per_sec: f32,
    pub unix_timestamp: u64,
}

#[derive(Clone)]
pub struct WebState {
    pub coordinator: Option<Coordinator>,
    pub loss_history: Vec<LossPoint>,
    /// Connected clients still downloading the checkpoint (not yet ready).
    pub syncing_clients: Vec<String>,
    /// Connected clients that have the checkpoint loaded and are awaiting
    /// epoch admission.
    pub ready_clients: Vec<String>,
    pub server_addr: String,
    pub wandb: Option<WandbInfo>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WandbInfo {
    pub project: String,
    pub run_name: String,
    pub entity: Option<String>,
    pub group: Option<String>,
}

pub(crate) type SharedState = Arc<Mutex<WebState>>;

pub fn start(
    state: WebState,
    port: u16,
    cancel: tokio_util::sync::CancellationToken,
) -> SharedState {
    let shared = Arc::new(Mutex::new(state));
    let app = Router::new()
        .route("/", get(index))
        .route("/partials/overview", get(overview_partial))
        .route("/partials/clients", get(clients_partial))
        .route("/partials/rounds", get(rounds_partial))
        .route("/partials/config", get(config_partial))
        .route("/partials/model", get(model_partial))
        .route("/partials/timing", get(timing_partial))
        .route("/partials/loss", get(loss_partial))
        .route("/partials/throughput", get(throughput_partial))
        .route("/api/state", get(api_state))
        .with_state(shared.clone());

    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
            .await
            .expect("Failed to bind web server");
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                cancel.cancelled().await;
            })
            .await
            .ok();
    });

    shared
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn overview_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let run_state = format_run_state(coord.run_state);
            let clients_count = coord.epoch_state.clients.len();
            let exited = coord.epoch_state.exited_clients.len();
            let step = coord.progress.step;
            let total_steps = coord.config.total_steps;
            let epoch = coord.progress.epoch;
            let height = coord.epoch_state.rounds[coord.epoch_state.rounds_head as usize].height;
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let state_duration = now.saturating_sub(coord.run_state_start_unix_timestamp);
            let epoch_duration = if coord.epoch_state.start_timestamp > 0 {
                format_duration(now.saturating_sub(coord.epoch_state.start_timestamp))
            } else {
                "-".into()
            };
            let pending = if coord.pending_pause.is_true() {
                "⚠️ Pending Pause"
            } else {
                ""
            };
            let cold_start = if coord.epoch_state.cold_start_epoch.is_true() {
                "Yes"
            } else {
                "No"
            };
            let first_round = if coord.epoch_state.first_round.is_true() {
                "Yes"
            } else {
                "No"
            };
            let syncing = s.syncing_clients.len();
            let ready = s.ready_clients.len();
            Html(format!(
                r#"<table border="1">
<tr><td><b>Run State</b></td><td>{run_state} {pending}</td></tr>
<tr><td><b>Run ID</b></td><td>{run_id}</td></tr>
<tr><td><b>Epoch</b></td><td>{epoch}</td></tr>
<tr><td><b>Step</b></td><td>{step} / {total_steps}</td></tr>
<tr><td><b>Height (Round)</b></td><td>{height}</td></tr>
<tr><td><b>Epoch Steps</b></td><td>{epoch_start_step} - {epoch_last_step}</td></tr>
<tr><td><b>Clients</b></td><td>{clients_count} ({exited} exited)</td></tr>
<tr><td><b>Connected</b></td><td><span style="color:#52b8cd">{ready} ready</span>, <span style="color:#e28844">{syncing} syncing</span></td></tr>
<tr><td><b>Server</b></td><td>{server_addr}</td></tr>
<tr><td><b>State Duration</b></td><td>{state_duration_str}</td></tr>
<tr><td><b>Epoch Duration</b></td><td>{epoch_duration}</td></tr>
<tr><td><b>Data Index</b></td><td>{data_index}</td></tr>
<tr><td><b>First Round</b></td><td>{first_round}</td></tr>
<tr><td><b>Cold Start Epoch</b></td><td>{cold_start}</td></tr>
</table>"#,
                run_state = run_state,
                pending = pending,
                run_id = coord.run_id,
                epoch = epoch,
                step = step,
                total_steps = total_steps,
                height = height,
                epoch_start_step = coord.epoch_state.start_step,
                epoch_last_step = coord.epoch_state.last_step,
                clients_count = clients_count,
                exited = exited,
                ready = ready,
                syncing = syncing,
                server_addr = s.server_addr,
                state_duration_str = format_duration(state_duration),
                epoch_duration = epoch_duration,
                data_index = coord.progress.epoch_start_data_index,
                first_round = first_round,
                cold_start = cold_start,
            ))
        }
        None => Html(r#"<i>Waiting for coordinator data...</i>"#.into()),
    }
}

async fn clients_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let mut healthy = 0u32;
            let mut dropped = 0u32;
            let mut withdrawn = 0u32;
            let mut ejected = 0u32;
            for client in coord.epoch_state.clients.iter() {
                match client.state {
                    ClientState::Healthy => healthy += 1,
                    ClientState::Dropped => dropped += 1,
                    ClientState::Withdrawn => withdrawn += 1,
                    ClientState::Ejected => ejected += 1,
                }
            }
            let total = healthy + dropped + withdrawn + ejected;

            let mut rows = String::new();
            for i in 0..coord.epoch_state.clients.len() {
                let client = &coord.epoch_state.clients[i];
                let id = client.id.to_string();
                let state_str = format!("{}", client.state);
                let exited = client.exited_height;
                let state_class = match client.state {
                    ClientState::Healthy => " style=\"color:#52b8cd\"",
                    ClientState::Dropped => " style=\"color:#da4e8a\"",
                    ClientState::Withdrawn => " style=\"color:#a85cbc\"",
                    ClientState::Ejected => " style=\"color:#da4e8a\"",
                };
                rows.push_str(&format!(
                    r#"<tr><td>{}</td><td{state_class}><b>{}</b></td><td>{}</td></tr>"#,
                    id, state_str, exited,
                ));
            }

            let mut exited_rows = String::new();
            for i in 0..coord.epoch_state.exited_clients.len() {
                let client = &coord.epoch_state.exited_clients[i];
                let id = client.id.to_string();
                let state_str = format!("{}", client.state);
                let exited = client.exited_height;
                exited_rows.push_str(&format!(
                    r#"<tr><td>{}</td><td><b>{}</b></td><td>{}</td></tr>"#,
                    id, state_str, exited,
                ));
            }

            if rows.is_empty() {
                rows = r#"<tr><td colspan="3"><i>No clients connected</i></td></tr>"#.into();
            }

            let exited_section = if !exited_rows.is_empty() {
                format!(
                    r#"<br><table border="1">
<caption><b>Exited Clients</b></caption>
<thead><tr><th>Client ID</th><th>Status</th><th>Exited Height</th></tr></thead>
<tbody>{exited_rows}</tbody>
</table>"#,
                )
            } else {
                String::new()
            };

            let connecting_rows: String = s
                .syncing_clients
                .iter()
                .map(|c| {
                    format!(
                        r#"<tr><td>{}</td><td style="color:#e28844"><b>Syncing</b></td></tr>"#,
                        c
                    )
                })
                .chain(s.ready_clients.iter().map(|c| {
                    format!(
                        r#"<tr><td>{}</td><td style="color:#e2ccb8"><b>Ready (waiting for next epoch)</b></td></tr>"#,
                        c
                    )
                }))
                .collect();

            let connecting_section = if !connecting_rows.is_empty() {
                format!(
                    r#"<br><table border="1">
<caption><b>Connected — Not Yet Admitted</b></caption>
<thead><tr><th>Client ID</th><th>Status</th></tr></thead>
<tbody>{connecting_rows}</tbody>
</table>"#,
                )
            } else {
                String::new()
            };

            Html(format!(
                r#"<div><b>Client Summary:</b> {total} total &mdash;
<span style="color:#52b8cd">{healthy} healthy</span>,
<span style="color:#da4e8a">{dropped} dropped</span>,
<span style="color:#a85cbc">{withdrawn} withdrawn</span>,
<span style="color:#da4e8a">{ejected} ejected</span></div>
<br>
<table border="1">
<thead><tr><th>Client ID</th><th>Status</th><th>Exited Height</th></tr></thead>
<tbody>{rows}</tbody>
</table>
{exited_section}
{connecting_section}"#,
                total = total,
                healthy = healthy,
                dropped = dropped,
                withdrawn = withdrawn,
                ejected = ejected,
                rows = rows,
            ))
        }
        None => Html(r#"<i>Waiting for coordinator data...</i>"#.into()),
    }
}

async fn rounds_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let head = coord.epoch_state.rounds_head as usize;
            let mut rows = String::new();
            for offset in 0..NUM_STORED_ROUNDS {
                let idx = (head + NUM_STORED_ROUNDS - offset) % NUM_STORED_ROUNDS;
                let round = &coord.epoch_state.rounds[idx];
                let is_current = offset == 0;
                let highlight = if is_current {
                    r#" style="font-weight:bold""#
                } else {
                    ""
                };
                let witness_count = round.witnesses.len();
                rows.push_str(&format!(
                    r#"<tr{highlight}>
<td>{height}</td>
<td>{data_index}</td>
<td>{random_seed}</td>
<td>{clients_len}</td>
<td>{tie_breaker_tasks}</td>
<td>{witness_count}</td>
</tr>"#,
                    height = round.height,
                    data_index = round.data_index,
                    random_seed = round.random_seed,
                    clients_len = round.clients_len,
                    tie_breaker_tasks = round.tie_breaker_tasks,
                    witness_count = witness_count,
                    highlight = highlight,
                ));
            }
            Html(format!(
                r#"<table border="1">
<thead><tr><th>Height</th><th>Data Index</th><th>Random Seed</th><th>Clients</th><th>TB Tasks</th><th>Witnesses</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<small>Bold row = current round (head)</small>"#,
                rows = rows,
            ))
        }
        None => Html(r#"<i>Waiting for coordinator data...</i>"#.into()),
    }
}

async fn config_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let cfg = &coord.config;
            Html(format!(
                r#"<table border="1">
<tr><td><b>Total Steps</b></td><td>{total_steps}</td></tr>
<tr><td><b>Epoch Time</b></td><td>{epoch_time}s</td></tr>
<tr><td><b>Warmup Time</b></td><td>{warmup_time}s</td></tr>
<tr><td><b>Cooldown Time</b></td><td>{cooldown_time}s</td></tr>
<tr><td><b>Max Round Train Time</b></td><td>{max_round_train_time}s</td></tr>
<tr><td><b>Round Witness Time</b></td><td>{round_witness_time}s</td></tr>
<tr><td><b>Warmup Batch Tokens</b></td><td>{warmup_tokens}</td></tr>
<tr><td><b>Init Min Clients</b></td><td>{init_min_clients}</td></tr>
<tr><td><b>Min Clients</b></td><td>{min_clients}</td></tr>
<tr><td><b>Witness Nodes</b></td><td>{witness_nodes}</td></tr>
<tr><td><b>Global Batch Size Start</b></td><td>{batch_start}</td></tr>
<tr><td><b>Global Batch Size End</b></td><td>{batch_end}</td></tr>
<tr><td><b>Verification %</b></td><td>{verification_percent}%</td></tr>
<tr><td><b>Waiting Extra Time</b></td><td>{waiting_extra}s</td></tr>
</table>"#,
                total_steps = cfg.total_steps,
                epoch_time = cfg.epoch_time,
                warmup_time = cfg.warmup_time,
                cooldown_time = cfg.cooldown_time,
                max_round_train_time = cfg.max_round_train_time,
                round_witness_time = cfg.round_witness_time,
                warmup_tokens = cfg.global_batch_size_warmup_tokens,
                init_min_clients = cfg.init_min_clients,
                min_clients = cfg.min_clients,
                witness_nodes = cfg.witness_nodes,
                batch_start = cfg.global_batch_size_start,
                batch_end = cfg.global_batch_size_end,
                verification_percent = cfg.verification_percent,
                waiting_extra = cfg.waiting_for_members_extra_time,
            ))
        }
        None => Html(r#"<i>Waiting for coordinator data...</i>"#.into()),
    }
}

fn format_lr_schedule(schedule: &LearningRateSchedule, current_step: u32) -> String {
    let schedule_type = match schedule {
        LearningRateSchedule::Constant(_) => "Constant",
        LearningRateSchedule::Linear(_) => "Linear",
        LearningRateSchedule::Cosine(_) => "Cosine",
        LearningRateSchedule::WarmupStableDecay(_) => "WarmupStableDecay",
    };
    let warmup_steps = schedule.get_warmup_steps();
    let warmup_init_lr = schedule.get_warmup_init_lr();
    let current_lr = schedule.get_lr(current_step);
    format!(
        "{} (warmup_steps={}, warmup_init_lr={:.8}, current_lr={:.8})",
        schedule_type, warmup_steps, warmup_init_lr, current_lr
    )
}

fn format_optimizer(opt: &OptimizerDefinition) -> String {
    match opt {
        OptimizerDefinition::Dummy => "Dummy".into(),
        OptimizerDefinition::AdamW {
            betas,
            weight_decay,
            eps,
            clip_grad_norm,
        } => {
            let clip = clip_grad_norm
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".into());
            format!(
                "AdamW(betas=[{},{}], weight_decay={}, eps={}, clip_grad_norm={})",
                betas[0], betas[1], weight_decay, eps, clip
            )
        }
        OptimizerDefinition::Distro {
            clip_grad_norm,
            weight_decay,
            compression_decay,
            compression_topk,
            compression_chunk,
            quantize_1bit,
        } => {
            let clip = clip_grad_norm
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".into());
            let wd = weight_decay
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".into());
            format!(
                "Distro(clip={}, wd={}, comp_decay={}, topk={}, chunk={}, quantize_1bit={})",
                clip, wd, compression_decay, compression_topk, compression_chunk, quantize_1bit
            )
        }
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

async fn model_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => match &coord.model {
            Model::LLM(llm) => {
                let arch = format_llm_architecture(&llm.architecture);
                let data_type = format_data_type(&llm.data_type);
                let cp_label = format_checkpoint_label(&llm.checkpoint);
                let lr_str = format_lr_schedule(&llm.lr_schedule, coord.progress.step);
                let opt_str = format_optimizer(&llm.optimizer);
                Html(format!(
                    r#"<table border="1">
<tr><td><b>Architecture</b></td><td>{arch}</td></tr>
<tr><td><b>Max Seq Len</b></td><td>{max_seq_len}</td></tr>
<tr><td><b>Cold Start Warmup Steps</b></td><td>{cold_start_steps}</td></tr>
<tr><td><b>Data Type</b></td><td>{data_type}</td></tr>
<tr><td><b>Checkpoint</b></td><td>{checkpoint}</td></tr>
<tr><td><b>LR Schedule</b></td><td>{lr_str}</td></tr>
<tr><td><b>Optimizer</b></td><td>{opt_str}</td></tr>
</table>"#,
                    arch = arch,
                    max_seq_len = llm.max_seq_len,
                    cold_start_steps = llm.cold_start_warmup_steps,
                    data_type = data_type,
                    checkpoint = cp_label,
                    lr_str = lr_str,
                    opt_str = opt_str,
                ))
            }
        },
        None => Html(r#"<i>Waiting for coordinator data...</i>"#.into()),
    }
}

fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn format_eta(secs: Option<u64>) -> String {
    secs.map(format_duration).unwrap_or_else(|| "-".into())
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

fn weighted_tokens_per_sec(points: &[&LossPoint]) -> Option<f64> {
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

    if total_weight > 0.0 {
        Some(weighted_total / total_weight)
    } else {
        None
    }
}

async fn timing_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let now = current_unix_timestamp();
            let points: Vec<&LossPoint> = s
                .loss_history
                .iter()
                .filter(|p| p.tokens_per_sec.is_finite() && p.tokens_per_sec > 0.0)
                .collect();
            let elapsed = points
                .first()
                .map(|p| now.saturating_sub(p.unix_timestamp))
                .or_else(|| {
                    (coord.run_state_start_unix_timestamp > 0)
                        .then(|| now.saturating_sub(coord.run_state_start_unix_timestamp))
                });

            let tokens_per_step = coord.get_target_global_batch_size(coord.current_round()) as f64
                * coord.get_sequence_length() as f64;
            let training_token_budget = coord.config.total_steps as u64
                * coord.config.global_batch_size_end as u64
                * coord.get_sequence_length() as u64;
            let remaining_tokens = training_token_budget
                .saturating_sub(coord.total_tokens_processed(coord.current_round()))
                as f64;
            let remaining_steps = (remaining_tokens / tokens_per_step.max(1.0)).ceil() as u32;
            let current_tps = points.last().map(|p| p.tokens_per_sec as f64);
            let avg_tps = if points.is_empty() {
                None
            } else {
                Some(
                    points.iter().map(|p| p.tokens_per_sec as f64).sum::<f64>()
                        / points.len() as f64,
                )
            };
            let weighted_tps = weighted_tokens_per_sec(&points);

            let avg_eta = estimate_remaining_time(remaining_tokens, avg_tps.unwrap_or(0.0));
            let current_eta = estimate_remaining_time(remaining_tokens, current_tps.unwrap_or(0.0));
            let weighted_eta =
                estimate_remaining_time(remaining_tokens, weighted_tps.unwrap_or(0.0));

            Html(format!(
                r#"<table border="1">
<tr><td><b>Elapsed</b></td><td>{elapsed}</td></tr>
<tr><td><b>Remaining Steps</b></td><td>{remaining_steps}</td></tr>
<tr><td><b>Remaining Tokens</b></td><td>{remaining_tokens}</td></tr>
<tr><td><b>Tokens / Step</b></td><td>{tokens_per_step:.0}</td></tr>
<tr><td><b>ETA (weighted avg)</b></td><td>{weighted_eta} <span class="hint">({weighted_tps})</span></td></tr>
<tr><td><b>ETA (overall avg)</b></td><td>{avg_eta} <span class="hint">({avg_tps})</span></td></tr>
<tr><td><b>ETA (current speed)</b></td><td>{current_eta} <span class="hint">({current_tps})</span></td></tr>
</table>"#,
                elapsed = format_eta(elapsed),
                remaining_steps = remaining_steps,
                remaining_tokens = format_tokens(remaining_tokens),
                tokens_per_step = tokens_per_step,
                weighted_eta = format_eta(weighted_eta),
                avg_eta = format_eta(avg_eta),
                current_eta = format_eta(current_eta),
                weighted_tps = format_tps(weighted_tps),
                avg_tps = format_tps(avg_tps),
                current_tps = format_tps(current_tps),
            ))
        }
        None => Html(r#"<i>Waiting for coordinator data...</i>"#.into()),
    }
}

fn format_tps(tps: Option<f64>) -> String {
    tps.map(|v| format!("{v:.1} tok/s"))
        .unwrap_or_else(|| "-".into())
}

fn format_tokens(tokens: f64) -> String {
    if tokens >= 1_000_000_000.0 {
        format!("{:.1}B", tokens / 1_000_000_000.0)
    } else if tokens >= 1_000_000.0 {
        format!("{:.1}M", tokens / 1_000_000.0)
    } else if tokens >= 1_000.0 {
        format!("{:.1}K", tokens / 1_000.0)
    } else {
        format!("{tokens:.0}")
    }
}

fn render_loss_svg(losses: &[LossPoint]) -> String {
    let width: f64 = 800.0;
    let height: f64 = 250.0;
    let pad_top = 20.0;
    let pad_bot = 30.0;
    let pad_left = 60.0;
    let pad_right = 20.0;
    let plot_x0 = pad_left;
    let plot_x1 = width - pad_right;
    let plot_y0 = pad_top;
    let plot_y1 = height - pad_bot;
    let plot_w = plot_x1 - plot_x0;
    let plot_h = plot_y1 - plot_y0;

    let filtered: Vec<&LossPoint> = losses.iter().filter(|l| l.loss.is_finite()).collect();
    let n = filtered.len();
    if n < 2 {
        return r#"<i>Not enough loss data yet</i>"#.into();
    }

    let tokens: Vec<u64> = filtered.iter().map(|l| l.tokens_processed).collect();
    let vals: Vec<f32> = filtered.iter().map(|l| l.loss).collect();

    let min_tokens = *tokens.first().unwrap_or(&0) as f64;
    let max_tokens = *tokens.last().unwrap_or(&1) as f64;
    let token_range = (max_tokens - min_tokens).max(1.0);

    let min_loss = vals.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
    let max_loss = vals
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let loss_range = (max_loss - min_loss).max(0.01) as f64;

    let mut points: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let x = plot_x0 + (tokens[i] as f64 - min_tokens) / token_range * plot_w;
        let y = plot_y1 - (vals[i] as f64 - min_loss as f64) / loss_range * plot_h;
        points.push(format!("{:.1},{:.1}", x, y));
    }
    let points_str = points.join(" ");

    let y_ticks = 5;
    let mut y_labels = String::new();
    for i in 0..=y_ticks {
        let val = min_loss as f64 + (max_loss - min_loss) as f64 * (i as f64 / y_ticks as f64);
        let y = plot_y1 - (i as f64 / y_ticks as f64) * plot_h;
        y_labels.push_str(&format!(
            r##"<text x="{}" y="{}" text-anchor="end" fill="#e2ccb8">{:.4}</text>"##,
            pad_left - 5.0,
            y + 4.0,
            val,
        ));
    }

    let x_ticks = 6;
    let mut x_labels = String::new();
    for i in 0..=x_ticks {
        let val = min_tokens + (max_tokens - min_tokens) * (i as f64 / x_ticks as f64);
        let x = plot_x0 + (i as f64 / x_ticks as f64) * plot_w;
        x_labels.push_str(&format!(
            r##"<text x="{x}" y="{}" text-anchor="middle" fill="#e2ccb8">{}</text>"##,
            height - 8.0,
            format_tokens(val),
        ));
    }

    let last_loss = vals.last().copied().unwrap_or(0.0);
    let avg_loss: f32 = vals.iter().copied().sum::<f32>() / vals.len() as f32;

    format!(
        r##"<div>
<b>Latest Loss:</b> {last_loss:.4} &nbsp; <b>Min:</b> {min_loss:.4} &nbsp; <b>Max:</b> {max_loss:.4} &nbsp; <b>Avg:</b> {avg_loss:.4} &nbsp; Tokens: {min_tokens} - {max_tokens} &nbsp; Points: {n}
<br><br>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg">
<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#463840"/>
{y_labels}
{x_labels}
<polyline points="{points_str}" fill="none" stroke="#e2ccb8"/>
</svg>
</div>"##,
        last_loss = last_loss,
        min_loss = min_loss,
        max_loss = max_loss,
        avg_loss = avg_loss,
        min_tokens = format_tokens(min_tokens),
        max_tokens = format_tokens(max_tokens),
        n = n,
        width = width,
        height = height,
        plot_x0 = plot_x0,
        plot_y0 = plot_y0,
        plot_w = plot_w,
        plot_h = plot_h,
        y_labels = y_labels,
        x_labels = x_labels,
        points_str = points_str,
    )
}

fn render_throughput_svg(losses: &[LossPoint]) -> String {
    let width: f64 = 800.0;
    let height: f64 = 200.0;
    let pad_top = 20.0;
    let pad_bot = 30.0;
    let pad_left = 70.0;
    let pad_right = 20.0;
    let plot_x0 = pad_left;
    let plot_x1 = width - pad_right;
    let plot_y0 = pad_top;
    let plot_y1 = height - pad_bot;
    let plot_w = plot_x1 - plot_x0;
    let plot_h = plot_y1 - plot_y0;

    let filtered: Vec<&LossPoint> = losses
        .iter()
        .filter(|l| l.tokens_per_sec.is_finite() && l.tokens_per_sec > 0.0)
        .collect();
    let n = filtered.len();
    if n < 2 {
        return r#"<i>Not enough throughput data yet</i>"#.into();
    }

    let tokens: Vec<u64> = filtered.iter().map(|l| l.tokens_processed).collect();
    let vals: Vec<f32> = filtered.iter().map(|l| l.tokens_per_sec).collect();

    let min_tokens = *tokens.first().unwrap_or(&0) as f64;
    let max_tokens = *tokens.last().unwrap_or(&1) as f64;
    let token_range = (max_tokens - min_tokens).max(1.0);

    let min_tps = vals.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
    let max_tps = vals
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let tps_range = (max_tps - min_tps).max(1.0) as f64;

    let mut points: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let x = plot_x0 + (tokens[i] as f64 - min_tokens) / token_range * plot_w;
        let y = plot_y1 - (vals[i] as f64 - min_tps as f64) / tps_range * plot_h;
        points.push(format!("{:.1},{:.1}", x, y));
    }
    let points_str = points.join(" ");

    let y_ticks = 4;
    let mut y_labels = String::new();
    for i in 0..=y_ticks {
        let val = min_tps as f64 + (max_tps - min_tps) as f64 * (i as f64 / y_ticks as f64);
        let y = plot_y1 - (i as f64 / y_ticks as f64) * plot_h;
        y_labels.push_str(&format!(
            r##"<text x="{}" y="{}" text-anchor="end" fill="#e2ccb8">{:.0}</text>"##,
            pad_left - 5.0,
            y + 4.0,
            val,
        ));
    }

    let x_ticks = 6;
    let mut x_labels = String::new();
    for i in 0..=x_ticks {
        let val = min_tokens + (max_tokens - min_tokens) * (i as f64 / x_ticks as f64);
        let x = plot_x0 + (i as f64 / x_ticks as f64) * plot_w;
        x_labels.push_str(&format!(
            r##"<text x="{x}" y="{}" text-anchor="middle" fill="#e2ccb8">{}</text>"##,
            height - 8.0,
            format_tokens(val),
        ));
    }

    let last_tps = vals.last().copied().unwrap_or(0.0);
    let avg_tps: f32 = vals.iter().copied().sum::<f32>() / vals.len() as f32;

    format!(
        r##"<div>
<b>Latest Tokens/s:</b> {last_tps:.1} &nbsp; <b>Avg:</b> {avg_tps:.1} &nbsp; <b>Peak:</b> {max_tps:.1} &nbsp; Tokens: {min_tokens} - {max_tokens} &nbsp; Points: {n}
<br><br>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart-svg">
<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#463840"/>
{y_labels}
{x_labels}
<polyline points="{points_str}" fill="none" stroke="#52b8cd"/>
</svg>
</div>"##,
        last_tps = last_tps,
        avg_tps = avg_tps,
        max_tps = max_tps,
        min_tokens = format_tokens(min_tokens),
        max_tokens = format_tokens(max_tokens),
        n = n,
        width = width,
        height = height,
        plot_x0 = plot_x0,
        plot_y0 = plot_y0,
        plot_w = plot_w,
        plot_h = plot_h,
        y_labels = y_labels,
        x_labels = x_labels,
        points_str = points_str,
    )
}

async fn loss_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    let svg = render_loss_svg(&s.loss_history);
    Html(svg)
}

async fn throughput_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    let svg = render_throughput_svg(&s.loss_history);
    Html(svg)
}

async fn api_state(State(state): State<SharedState>) -> impl axum::response::IntoResponse {
    let s = state.lock().unwrap();
    let json = serde_json::json!({
        "coordinator": &s.coordinator,
        "loss_history": &s.loss_history,
        "syncing_clients": &s.syncing_clients,
        "ready_clients": &s.ready_clients,
        "server_addr": &s.server_addr,
        "wandb": &s.wandb,
    });
    axum::Json(json)
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

fn format_data_type(dt: &LLMTrainingDataType) -> &'static str {
    match dt {
        LLMTrainingDataType::Pretraining => "Pretraining",
        LLMTrainingDataType::Finetuning => "Finetuning",
    }
}

const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html>
<head>
<title>Aether Training Monitor</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>
body { font-family: monospace; margin: 0; background: #141216; color: #e2ccb8; font-size: 13px; }
.wrap { max-width: 1200px; margin: 0 auto; padding: 0 1rem 1.5rem; }
.topbar { position: sticky; top: 0; z-index: 10; background: #141216; border-bottom: 1px solid #463840; padding: .55rem 0; }
.topbar .wrap { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.topbar h1 { font-size: 15px; margin: 0; }
.hint { color: #746268; font-size: 12px; }
h1 { color: #f5ead9; }
h2 { color: #e2ccb8; border-bottom: 1px solid #463840; padding-bottom: 4px; font-size: 14px; margin: 0 0 .5rem; }
table { border-collapse: collapse; }
td, th { padding: 3px 8px; text-align: left; border: 1px solid #463840; font-size: 12px; }
b { color: #f5ead9; }
a { color: #52b8cd; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem 1.25rem; margin-top: 1rem; align-items: start; }
svg { display: block; width: 100%; height: auto; }
.chart-svg text { fill: #e2ccb8; font-size: 11px; }
.panel { overflow-x: auto; }
</style>
</head>
<body>
<header class="topbar"><div class="wrap">
  <h1>Aether Training Monitor</h1>
  <span class="hint">live &middot; auto-refresh 2&ndash;5s</span>
</div></header>
<div class="wrap">
<div class="grid">
  <div class="panel">
    <h2>Overview</h2>
    <div hx-get="/partials/overview" hx-trigger="every 2s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
  <div class="panel">
    <h2>Configuration</h2>
    <div hx-get="/partials/config" hx-trigger="every 5s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
  <div class="panel">
    <h2>Model</h2>
    <div hx-get="/partials/model" hx-trigger="every 5s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
</div>
<div class="grid">
  <div class="panel">
    <h2>Clients</h2>
    <div hx-get="/partials/clients" hx-trigger="every 2s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
  <div class="panel">
    <h2>Rounds</h2>
    <div hx-get="/partials/rounds" hx-trigger="every 3s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
  <div class="panel">
    <h2>Timing</h2>
    <div hx-get="/partials/timing" hx-trigger="every 2s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
</div>
<div class="grid">
  <div class="panel">
    <h2>Loss</h2>
    <div hx-get="/partials/loss" hx-trigger="every 5s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
  <div class="panel">
    <h2>Throughput (Tokens/sec)</h2>
    <div hx-get="/partials/throughput" hx-trigger="every 5s" hx-swap="innerHTML"><i>Loading...</i></div>
  </div>
</div>
</div>
</body>
</html>"##;
