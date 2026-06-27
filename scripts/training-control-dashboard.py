#!/usr/bin/env python3
import html
import json
import os
import base64
import shlex
import signal
import subprocess
import sys
import threading
import time
import tomllib
import secrets
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


CONFIG_PATH = Path(os.environ.get("TRAINING_RUN_CONFIG", "config/training-run.toml"))
CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "8080"))
CONTROL_USERNAME = os.environ.get("CONTROL_USERNAME", "admin")
CONTROL_PASSWORD = os.environ.get("CONTROL_PASSWORD") or secrets.token_urlsafe(24)
GENERATED_CONTROL_PASSWORD = "CONTROL_PASSWORD" not in os.environ
MAX_LOG_LINES = 500


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (repo_root() / CONFIG_PATH).open("rb") as f:
        return tomllib.load(f)


def bool_value(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def int_value(value: str) -> int:
    return int(value.strip())


def write_config(config: dict) -> None:
    lines: list[str] = []
    for section in ("server", "dataset", "model"):
        lines.append(f"[{section}]")
        for key, value in config.get(section, {}).items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, int):
                lines.append(f"{key} = {value}")
            else:
                escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
                lines.append(f'{key} = "{escaped}"')
        lines.append("")
    (repo_root() / CONFIG_PATH).write_text("\n".join(lines), encoding="utf-8")


def update_config_from_form(form: dict[str, list[str]]) -> None:
    config = load_config()
    schema = {
        "server": {
            "state_path": str,
            "server_port": int_value,
            "live_web_port": int_value,
            "tui": bool_value,
        },
        "dataset": {
            "enabled": bool_value,
            "script": str,
            "output_dir": str,
            "sequence_length": int_value,
            "num_sequences": int_value,
            "shard_size": int_value,
            "token_bytes": int_value,
            "dataset": str,
            "split": str,
            "text_field": str,
            "tokenizer": str,
            "seed": int_value,
            "buffer_docs": int_value,
            "trust_remote_code": bool_value,
        },
        "model": {
            "enabled": bool_value,
            "script": str,
            "config": str,
            "repo": str,
            "tokenizer": str,
            "private": bool_value,
            "dtype": str,
            "device": str,
        },
    }
    for section, keys in schema.items():
        config.setdefault(section, {})
        for key, converter in keys.items():
            field_name = f"{section}.{key}"
            if converter is bool_value:
                config[section][key] = field_name in form
            elif field_name in form:
                config[section][key] = converter(form[field_name][0])
    write_config(config)


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def marker_dir() -> Path:
    path = repo_root() / ".aether-control"
    path.mkdir(exist_ok=True)
    return path


def model_marker_path() -> Path:
    return marker_dir() / "model-push.json"


def dataset_status(config: dict) -> tuple[bool, str]:
    dataset = config.get("dataset", {})
    output_dir = repo_root() / dataset.get("output_dir", "")
    metadata_path = output_dir / "subset_metadata.json"
    if not output_dir.exists():
        return False, f"missing {output_dir}"
    if not metadata_path.exists():
        return False, f"missing {metadata_path}"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        return False, f"invalid metadata: {err}"
    expected_sequences = int(dataset.get("num_sequences", 0))
    actual_sequences = int(metadata.get("num_sequences", 0))
    if actual_sequences < expected_sequences:
        return False, f"has {actual_sequences:,}/{expected_sequences:,} sequences"
    expected_seq_len = int(dataset.get("sequence_length", 0))
    if int(metadata.get("sequence_length", 0)) != expected_seq_len:
        return False, "sequence length does not match config"
    expected_token_bytes = int(dataset.get("token_bytes", 0))
    if int(metadata.get("token_bytes", 0)) != expected_token_bytes:
        return False, "token byte width does not match config"
    shard_count = len(list(output_dir.glob("*.bin")))
    if shard_count == 0:
        return False, "metadata exists but no .bin shards were found"
    return True, f"ready: {actual_sequences:,} sequences across {shard_count:,} shards"


def model_status(config: dict) -> tuple[bool, str]:
    model = config.get("model", {})
    if not model.get("enabled", True):
        return True, "disabled"
    marker = model_marker_path()
    if not marker.exists():
        return False, "no successful push recorded by this dashboard"
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "push marker is invalid"
    repo = model.get("repo", "")
    if data.get("repo") != repo:
        return False, f"last push was for {data.get('repo', '<unknown>')}"
    return True, f"last pushed {time.ctime(data.get('timestamp', 0))}"


def state_checkpoint(config: dict) -> str:
    state_path = repo_root() / config.get("server", {}).get("state_path", "")
    if not state_path.exists():
        return "state file missing"
    text = state_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("repo_id"):
            return line.split("=", 1)[1].strip().strip('"')
    return "checkpoint repo not found in state file"


@dataclass
class Job:
    name: str
    process: subprocess.Popen | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None
    command: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class ControlState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job: Job | None = None
        self.server: Job | None = None

    def append_log(self, job: Job, line: str) -> None:
        with self.lock:
            job.log.append(line.rstrip())
            del job.log[:-MAX_LOG_LINES]


STATE = ControlState()


def run_background(name: str, command: list[str], on_success=None, long_running: bool = False) -> Job:
    with STATE.lock:
        active = STATE.server if long_running else STATE.job
        if active and active.running:
            raise RuntimeError(f"{active.name} is already running")
        job = Job(name=name, command=command)
        target_attr = "server" if long_running else "job"
        setattr(STATE, target_attr, job)

    def worker() -> None:
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=repo_root(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with STATE.lock:
                job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                STATE.append_log(job, line)
            returncode = process.wait()
            with STATE.lock:
                job.returncode = returncode
                job.finished_at = time.time()
            if returncode == 0 and on_success is not None:
                on_success()
        except Exception as exc:
            with STATE.lock:
                job.log.append(f"ERROR: {exc}")
                job.returncode = -1
                job.finished_at = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return job


def prepare_dataset_command(config: dict) -> list[str]:
    dataset = config["dataset"]
    command = [
        sys.executable,
        dataset.get("script", "scripts/prepare-ultra-fineweb-local.py"),
        "--dataset",
        dataset["dataset"],
        "--split",
        dataset["split"],
        "--text-field",
        dataset["text_field"],
        "--tokenizer",
        dataset["tokenizer"],
        "--output-dir",
        dataset["output_dir"],
        "--sequence-length",
        str(dataset["sequence_length"]),
        "--num-sequences",
        str(dataset["num_sequences"]),
        "--shard-size",
        str(dataset["shard_size"]),
        "--token-bytes",
        str(dataset["token_bytes"]),
        "--seed",
        str(dataset["seed"]),
        "--buffer-docs",
        str(dataset["buffer_docs"]),
    ]
    if dataset.get("subset"):
        command.extend(["--subset", dataset["subset"]])
    if dataset.get("trust_remote_code", False):
        command.append("--trust-remote-code")
    return command


def push_model_command(config: dict) -> list[str]:
    model = config["model"]
    command = [
        sys.executable,
        model.get("script", "scripts/push-new-model-hf.py"),
        "--config",
        model["config"],
        "--repo",
        model["repo"],
        "--tokenizer",
        model["tokenizer"],
    ]
    if model.get("private", False):
        command.append("--private")
    if model.get("device"):
        command.extend(["--device", model["device"]])
    dtype = model.get("dtype", "")
    if dtype and dtype != "bfloat16":
        command.extend(["--dtype", dtype])
    return command


def validate_command(config: dict) -> list[str]:
    return [
        "psyche-centralized-server",
        "validate-config",
        "--state",
        config["server"]["state_path"],
    ]


def server_command(config: dict) -> list[str]:
    server = config["server"]
    command = [
        "psyche-centralized-server",
        "run",
        "--state",
        server["state_path"],
        "--server-port",
        str(server["server_port"]),
        "--web-port",
        str(server["live_web_port"]),
    ]
    if not server.get("tui", False):
        command.extend(["--tui=false"])
    return command


TAB_SCRIPT = """
<script>
const t = document.querySelectorAll('.tabs button.tab');
const p = document.querySelectorAll('[data-panel]');
t.forEach(b => b.addEventListener('click', () => {
  t.forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  p.forEach(s => { s.hidden = s.dataset.panel !== b.dataset.tab; });
}));
</script>
"""


def html_page(message: str | None = None) -> str:
    config = load_config()
    data_ready, data_message = dataset_status(config)
    model_ready, model_message = model_status(config)
    checkpoint = state_checkpoint(config)
    with STATE.lock:
        job = STATE.job
        server = STATE.server
    if server and server.running:
        server_short, server_cls = "running", "ok"
    elif server is None:
        server_short, server_cls = "idle", "warn"
    elif server.returncode == 0:
        server_short, server_cls = "stopped", "ok"
    else:
        server_short, server_cls = "stopped", "bad"
    data_short = "ready" if data_ready else "pending"
    data_cls = "ok" if data_ready else "bad"
    model_short = "ready" if model_ready else "pending"
    model_cls = "ok" if model_ready else "warn"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aether Training Control</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; font-size: 13px; line-height: 1.45; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 1rem 2rem; }}
    .topbar {{ position: sticky; top: 0; z-index: 10; background: #0d1117; border-bottom: 1px solid #30363d; }}
    .topbar .wrap {{ display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap; padding-top: .6rem; padding-bottom: .6rem; }}
    .brand {{ font-size: 15px; font-weight: 700; color: #f0f6fc; }}
    .statusline {{ font-size: 12px; color: #8b949e; }}
    .msg {{ padding: .5rem .75rem; margin: 1rem 0 0; border: 1px solid #30363d; }}
    .tabs {{ display: flex; border-bottom: 1px solid #30363d; margin-top: 1rem; }}
    .tabs button.tab {{ all: unset; cursor: pointer; padding: .45rem .8rem; color: #8b949e; border-bottom: 2px solid transparent; font: inherit; }}
    .tabs button.tab:hover {{ color: #e6edf3; background: transparent; }}
    .tabs button.tab.active {{ color: #f0f6fc; border-bottom-color: #58a6ff; background: transparent; }}
    [data-panel] {{ margin-top: 1rem; }}
    input[type="text"], input:not([type]) {{ width: 100%; box-sizing: border-box; padding: .3rem; background: #161b22; color: #e6edf3; border: 1px solid #30363d; font: inherit; }}
    input[type="checkbox"] {{ accent-color: #58a6ff; }}
    label {{ display: block; font-weight: 600; margin-top: .55rem; font-size: 12px; }}
    fieldset {{ border: 1px solid #30363d; margin: 1rem 0; padding: .75rem; }}
    legend {{ color: #8b949e; padding: 0 .35rem; }}
    button {{ padding: .4rem .7rem; background: #21262d; color: #e6edf3; border: 1px solid #30363d; cursor: pointer; font: inherit; }}
    button:hover {{ background: #2d333b; }}
    button.primary {{ background: #1f6feb; border-color: #1f6feb; color: #fff; }}
    .ok {{ color: #3fb950; font-weight: 700; }}
    .warn {{ color: #d29922; font-weight: 700; }}
    .bad {{ color: #f85149; font-weight: 700; }}
    pre {{ background: #161b22; color: #e6edf3; padding: .75rem; overflow: auto; max-height: 22rem; border: 1px solid #30363d; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: .5rem .75rem; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: .5rem; }}
    .actions form {{ margin: 0; }}
    h3 {{ margin: 1rem 0 .4rem; font-size: 13px; color: #c9d1d9; }}
    code {{ color: #f0f6fc; font-size: 12px; }}
    a {{ color: #58a6ff; }}
    p {{ margin: .3rem 0; }}
  </style>
</head>
<body>
  <header class="topbar"><div class="wrap">
    <span class="brand">Aether Training Control</span>
    <span class="statusline">Dataset: <span class="{data_cls}">{data_short}</span> &middot; Init model: <span class="{model_cls}">{model_short}</span> &middot; Server: <span class="{server_cls}">{server_short}</span></span>
  </div></header>
  <main class="wrap">
    {f'<div class="msg warn">{html.escape(message)}</div>' if message else ''}
    <nav class="tabs">
      <button type="button" class="tab active" data-tab="status">Status</button>
      <button type="button" class="tab" data-tab="config">Config</button>
      <button type="button" class="tab" data-tab="actions">Actions</button>
      <button type="button" class="tab" data-tab="logs">Logs</button>
    </nav>
    <section data-panel="status">
      <p>Dataset: <span class="{'ok' if data_ready else 'bad'}">{html.escape(data_message)}</span></p>
      <p>Init model: <span class="{'ok' if model_ready else 'warn'}">{html.escape(model_message)}</span></p>
      <p>State checkpoint: <code>{html.escape(checkpoint)}</code></p>
      <p>Training server: {render_job_status(server, live=True)}</p>
      <p>Live dashboard: <a href="http://{html.escape(os.environ.get('PUBLIC_HOST', 'localhost'))}:{config['server']['live_web_port']}/">port {config['server']['live_web_port']}</a></p>
    </section>
    <section data-panel="config" hidden>
      <form method="post" action="/save">
        {render_config_form(config)}
        <button class="primary" type="submit">Save configuration</button>
      </form>
    </section>
    <section data-panel="actions" hidden>
      <div class="actions">
        <form method="post" action="/prepare-dataset"><button type="submit">Prepare dataset</button></form>
        <form method="post" action="/push-model"><button type="submit">Push init model</button></form>
        <form method="post" action="/validate"><button type="submit">Validate state config</button></form>
        <form method="post" action="/start-server"><button type="submit">Start training server</button></form>
        <form method="post" action="/stop-server"><button type="submit">Stop training server</button></form>
      </div>
    </section>
    <section data-panel="logs" hidden>
      <h3>Last Job</h3>
      {render_job(job)}
      <h3>Server Log</h3>
      {render_job(server)}
    </section>
  </main>
  {TAB_SCRIPT}
</body>
</html>"""


def render_job_status(job: Job | None, live: bool = False) -> str:
    if job is None:
        return "not started"
    if job.running:
        return f'<span class="ok">running</span> <code>{html.escape(shell_join(job.command))}</code>'
    css = "ok" if job.returncode == 0 else "bad"
    noun = "stopped" if live else "finished"
    return f'<span class="{css}">{noun} ({job.returncode})</span>'


def render_job(job: Job | None) -> str:
    if job is None:
        return "<p>No job has run yet.</p>"
    lines = "\n".join(html.escape(line) for line in job.log)
    return f"<p>{render_job_status(job)}</p><p><code>{html.escape(shell_join(job.command))}</code></p><pre>{lines}</pre>"


def render_config_form(config: dict) -> str:
    sections = []
    for section, values in config.items():
        fields = []
        for key, value in values.items():
            name = f"{section}.{key}"
            label = html.escape(name)
            if isinstance(value, bool):
                checked = " checked" if value else ""
                fields.append(f'<label><input style="width:auto" type="checkbox" name="{label}"{checked}> {label}</label>')
            else:
                fields.append(f'<label>{label}<input name="{label}" value="{html.escape(str(value))}"></label>')
        sections.append(f"<fieldset><legend>{html.escape(section)}</legend><div class=\"grid\">{''.join(fields)}</div></fieldset>")
    return "".join(sections)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        if not self.authorized():
            self.request_auth()
            return
        self.respond(html_page())

    def do_POST(self) -> None:
        if not self.authorized():
            self.request_auth()
            return
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        path = urlparse(self.path).path
        message = None
        try:
            config = load_config()
            if path == "/save":
                update_config_from_form(form)
                message = "Configuration saved."
            elif path == "/prepare-dataset":
                command = prepare_dataset_command(config)
                run_background("prepare dataset", command)
                message = "Dataset preparation started."
            elif path == "/push-model":
                command = push_model_command(config)

                def mark_model() -> None:
                    model_marker_path().write_text(
                        json.dumps({"repo": config["model"]["repo"], "timestamp": time.time()}, indent=2) + "\n",
                        encoding="utf-8",
                    )

                run_background("push init model", command, on_success=mark_model)
                message = "Init model push started."
            elif path == "/validate":
                run_background("validate config", validate_command(config))
                message = "Config validation started."
            elif path == "/start-server":
                data_ready, data_message = dataset_status(config)
                if config.get("dataset", {}).get("enabled", True) and not data_ready:
                    raise RuntimeError(f"dataset is not ready: {data_message}")
                model_ready, model_message = model_status(config)
                if config.get("model", {}).get("enabled", True) and not model_ready:
                    raise RuntimeError(f"init model is not ready: {model_message}")
                run_background("training server", server_command(config), long_running=True)
                message = "Training server started."
            elif path == "/stop-server":
                with STATE.lock:
                    server = STATE.server
                if server and server.running and server.process:
                    server.process.send_signal(signal.SIGTERM)
                    message = "Stop signal sent."
                else:
                    message = "Training server is not running."
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        except Exception as err:
            message = str(err)
        self.respond(html_page(message))

    def respond(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self) -> bool:
        header = self.headers.get("authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header.removeprefix("Basic ")).decode("utf-8")
        except Exception:
            return False
        username, _, password = decoded.partition(":")
        return secrets.compare_digest(username, CONTROL_USERNAME) and secrets.compare_digest(
            password, CONTROL_PASSWORD
        )

    def request_auth(self) -> None:
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("www-authenticate", 'Basic realm="Aether Training Control"')
        self.send_header("content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Authentication required\n")

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def main() -> None:
    os.chdir(repo_root())
    server = ThreadingHTTPServer(("0.0.0.0", CONTROL_PORT), Handler)
    print(f"training control dashboard listening on 0.0.0.0:{CONTROL_PORT}", flush=True)
    if GENERATED_CONTROL_PASSWORD:
        print(
            f"generated control dashboard credentials: {CONTROL_USERNAME}:{CONTROL_PASSWORD}",
            flush=True,
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
