use std::{
    io::{self, Write},
    ops::{Deref, DerefMut},
    panic,
};

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::{Backend, CrosstermBackend},
    Terminal,
};
use tracing::{error, trace};

pub struct TerminalWrapper<T: Backend>(pub Terminal<T>);

impl<T: Backend> Deref for TerminalWrapper<T> {
    type Target = Terminal<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Backend> DerefMut for TerminalWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub fn init_terminal() -> io::Result<TerminalWrapper<impl Backend>> {
    trace!(target:"crossterm", "Initializing terminal");
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    terminal.hide_cursor()?;

    // Restore the terminal before printing panic messages.
    let default_panic = std::panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        restore_terminal();
        let _ = io::stdout().flush();
        default_panic(info);
        let _ = io::stdout().flush();
    }));

    Ok(TerminalWrapper(terminal))
}

impl<T: Backend> Drop for TerminalWrapper<T> {
    fn drop(&mut self) {
        restore_terminal();
    }
}

fn restore_terminal() {
    trace!(target:"crossterm", "Restoring terminal");
    if let Err(err) = disable_raw_mode() {
        error!("failed to disable terminal raw mode: {err:#}");
    }
    if let Err(err) = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture) {
        error!("failed to leave alternate screen & disable mouse capture: {err:#}");
    }
}
