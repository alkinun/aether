use axum::{
    extract::State,
    response::Html,
    routing::get,
    Router,
};
use psyche_coordinator::{Coordinator, RunState};
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Serialize)]
pub struct LossPoint {
    pub step: u32,
    pub loss: f32,
    pub tokens_per_sec: f32,
}

#[derive(Clone)]
pub struct WebState {
    pub coordinator: Option<Coordinator>,
    pub loss_history: Vec<LossPoint>,
    pub pending_clients: Vec<String>,
    pub server_addr: String,
}

type SharedState = Arc<Mutex<WebState>>;

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
        .route("/partials/loss", get(loss_partial))
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
            Html(format!(
                r#"<table border="1">
<tr><td><b>Run State</b></td><td>{run_state}</td></tr>
<tr><td><b>Step</b></td><td>{step} / {total_steps}</td></tr>
<tr><td><b>Epoch</b></td><td>{epoch}</td></tr>
<tr><td><b>Height (Round)</b></td><td>{height}</td></tr>
<tr><td><b>Clients</b></td><td>{clients_count} ({exited} exited)</td></tr>
<tr><td><b>Server</b></td><td>{server_addr}</td></tr>
</table>"#,
                run_state = run_state,
                step = step,
                total_steps = total_steps,
                epoch = epoch,
                height = height,
                clients_count = clients_count,
                exited = exited,
                server_addr = s.server_addr,
            ))
        }
        None => Html(
            r#"<i>Waiting for coordinator data...</i>"#.into(),
        ),
    }
}

async fn clients_partial(State(state): State<SharedState>) -> Html<String> {
    let s = state.lock().unwrap();
    match &s.coordinator {
        Some(coord) => {
            let mut rows = String::new();
            for i in 0..coord.epoch_state.clients.len() {
                let client = &coord.epoch_state.clients[i];
                let id = client.id.to_string();
                let state_str = format!("{}", client.state);
                let exited = client.exited_height;
                rows.push_str(&format!(
                    r#"<tr><td>{}</td><td><b>{}</b></td><td>{}</td></tr>"#,
                    id, state_str, exited,
                ));
            }
            if rows.is_empty() {
                rows = r#"<tr><td colspan="3"><i>No clients connected</i></td></tr>"#.into();
            }
            Html(format!(
                r#"<table border="1">
<thead><tr><th>Client ID</th><th>Status</th><th>Exited Height</th></tr></thead>
<tbody>{rows}</tbody>
</table>"#,
                rows = rows,
            ))
        }
        None => Html(
            r#"<i>Waiting for coordinator data...</i>"#.into(),
        ),
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

    let steps: Vec<u32> = filtered.iter().map(|l| l.step).collect();
    let vals: Vec<f32> = filtered.iter().map(|l| l.loss).collect();

    let min_step = *steps.first().unwrap_or(&0) as f64;
    let max_step = *steps.last().unwrap_or(&1) as f64;
    let step_range = (max_step - min_step).max(1.0);

    let min_loss = vals.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
    let max_loss = vals.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let loss_range = (max_loss - min_loss).max(0.01) as f64;

    let mut points: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let x = plot_x0 + (steps[i] as f64 - min_step) / step_range * plot_w;
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
            r##"<text x="{}" y="{}" text-anchor="end">{:.2}</text>"##,
            pad_left - 5.0,
            y + 4.0,
            val,
        ));
    }

    let x_ticks = 6;
    let mut x_labels = String::new();
    for i in 0..=x_ticks {
        let val = min_step + (max_step - min_step) * (i as f64 / x_ticks as f64);
        let x = plot_x0 + (i as f64 / x_ticks as f64) * plot_w;
        x_labels.push_str(&format!(
            r##"<text x="{x}" y="{}" text-anchor="middle">{:.0}</text>"##,
            height - 8.0,
            val,
        ));
    }

    let last_loss = vals.last().copied().unwrap_or(0.0);

    format!(
        r##"<div>
<b>Latest Loss:</b> {last_loss:.4} &nbsp;&nbsp; Steps: {min_step:.0} - {max_step:.0}
<br><br>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black"/>
{y_labels}
{x_labels}
<polyline points="{points_str}" fill="none" stroke="black"/>
</svg>
</div>"##,
        last_loss = last_loss,
        min_step = min_step,
        max_step = max_step,
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

async fn api_state(State(state): State<SharedState>) -> impl axum::response::IntoResponse {
    let s = state.lock().unwrap();
    let json = serde_json::json!({
        "coordinator": &s.coordinator,
        "loss_history": &s.loss_history,
        "pending_clients": &s.pending_clients,
        "server_addr": &s.server_addr,
    });
    axum::Json(json)
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

const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html>
<head>
<title>Psyche Monitor</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
</head>
<body>

<h1>Psyche Monitor</h1>
<hr>

<table>
<tr valign="top">
<td>
<h2>Overview</h2>
<div hx-get="/partials/overview" hx-trigger="every 2s" hx-swap="innerHTML"><i>Loading...</i></div>
</td>
<td>
<h2>Clients</h2>
<div hx-get="/partials/clients" hx-trigger="every 2s" hx-swap="innerHTML"><i>Loading...</i></div>
</td>
</tr>
</table>

<hr>

<h2>Loss</h2>
<div hx-get="/partials/loss" hx-trigger="every 5s" hx-swap="innerHTML"><i>Loading...</i></div>

</body>
</html>"##;
