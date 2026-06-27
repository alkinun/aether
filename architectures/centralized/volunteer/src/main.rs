//! aether-volunteer — branded launcher that onboards a volunteer and then
//! execs the real `psyche-centralized-client` training binary.
//!
//! Keep this crate torch-free and dependency-light so it compiles in seconds
//! (it's the first thing the installer builds). The heavy client is compiled
//! lazily from the TUI's "prepare" screen while the volunteer watches.

mod app;
mod brand;
mod config;
mod detect;
mod logo;
mod prepare;
mod requirements;
mod terminal;

use anyhow::Result;

fn main() -> Result<()> {
    // Drive the whole onboarding TUI. The terminal (alt screen / raw mode) is
    // acquired and released *inside* `run`, so by the time we get a `Launch`
    // config back the user's terminal is back to normal and safe to hand off
    // to the training client via exec.
    let launch = match app::run()? {
        Some(launch) => launch,
        None => return Ok(()), // user quit during onboarding
    };

    // Replace this process with the training client. The client re-initializes
    // its own terminal (ratatui) for its dashboard.
    prepare::exec_client(&launch)?;

    Ok(())
}
