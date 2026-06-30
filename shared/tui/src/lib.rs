mod app;
pub mod logging;
mod maybe;
mod tabbed;
pub mod terminal;
mod widget;

use anyhow::Result;
use tokio::{
    signal,
    sync::mpsc::{self, Sender},
};
use tokio_util::sync::CancellationToken;
use tracing::error;

pub use app::App;
pub use logging::{logging, LogOutput, LoggerWidget, ServiceInfo};
pub use maybe::MaybeTui;
pub use tabbed::TabbedWidget;
pub use terminal::{init_terminal, TerminalWrapper};
pub use widget::CustomWidget;

pub fn start_render_loop<T: CustomWidget>(
    widget: T,
) -> Result<(CancellationToken, Sender<T::Data>)> {
    let (tx, rx) = mpsc::channel(10);
    let cancel = CancellationToken::new();
    let terminal = init_terminal()?;

    tokio::spawn({
        let cancel = cancel.clone();
        async move {
            if let Err(error) = App::new(widget).start(cancel.clone(), terminal, rx).await {
                error!("TUI render loop failed: {error:#}");
                cancel.cancel();
            }
        }
    });
    Ok((cancel, tx))
}

pub fn maybe_start_render_loop<T: CustomWidget>(
    widget: Option<T>,
) -> Result<(CancellationToken, Option<Sender<T::Data>>)> {
    Ok(match widget {
        Some(widget) => {
            let (cancel, tx) = start_render_loop(widget)?;
            (cancel, Some(tx))
        }
        None => (setup_ctrl_c(), None),
    })
}

pub fn setup_ctrl_c() -> CancellationToken {
    let token = CancellationToken::new();
    tokio::spawn({
        let token = token.clone();
        async move {
            match signal::ctrl_c().await {
                Ok(()) => token.cancel(),
                Err(error) => error!("failed to listen for ctrl-c: {error:#}"),
            }
        }
    });
    token
}

pub use crossterm;
pub use ratatui;
