//! Minimal crossterm + ratatui terminal setup. We intentionally do NOT pull in
//! `psyche-tui` here (it drags opentelemetry/iroh/logfire) so this crate stays
//! tiny and fast to compile.

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, Stdout};

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

pub struct TerminalGuard {
    term: Tui,
    restored: bool,
}

impl TerminalGuard {
    pub fn init() -> io::Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(io::stdout());
        let term = Terminal::new(backend)?;
        Ok(Self {
            term,
            restored: false,
        })
    }

    pub fn term(&mut self) -> &mut Tui {
        &mut self.term
    }

    /// Restore the terminal explicitly. Needed because we hand control to the
    /// training client via `exec`, which bypasses `Drop`.
    pub fn restore(&mut self) {
        if self.restored {
            return;
        }
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        self.restored = true;
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}
