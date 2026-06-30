use crate::{terminal::TerminalWrapper, widget::CustomWidget};
use crossterm::event::{Event, EventStream, KeyCode, KeyModifiers};
use futures::StreamExt;
use ratatui::{backend::Backend, Terminal};
use std::time::Duration;
use tokio::{select, sync::mpsc::Receiver};
use tokio_util::sync::CancellationToken;
use tracing::{debug, trace};

pub struct App<W: CustomWidget> {
    custom_widget: W,
    custom_widget_data_state: W::Data,
}

impl<W: CustomWidget> App<W> {
    pub fn new(widget: W) -> Self {
        Self {
            custom_widget: widget,
            custom_widget_data_state: Default::default(),
        }
    }

    pub async fn start(
        mut self,
        shutdown_token: CancellationToken,
        mut terminal: TerminalWrapper<impl Backend>,
        mut state_rx: Receiver<W::Data>,
    ) -> anyhow::Result<()> {
        let mut frame_interval = tokio::time::interval(Duration::from_millis(150));
        let mut reader = EventStream::new();

        loop {
            select! {
                _ = shutdown_token.cancelled() => {
                    break;
                }
                _ = frame_interval.tick() => {
                    self.draw(&mut terminal)?;
                }
                Some(Ok(event)) = reader.next() => {
                    trace!(target:"crossterm", "Stdin event received {:?}", event);
                    self.handle_ui_event(event, &shutdown_token);
                    self.draw(&mut terminal)?;
                }
                Some(state) = state_rx.recv() => {
                    self.custom_widget_data_state = state;
                    self.draw(&mut terminal)?;
                }
            }
        }
        Ok(())
    }

    fn handle_ui_event(&mut self, event: Event, shutdown_token: &CancellationToken) {
        debug!(target: "App", "Handling UI event: {:?}",event);

        self.custom_widget.on_ui_event(&event);

        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Char('q') => shutdown_token.cancel(),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    shutdown_token.cancel()
                }
                _ => {}
            }
        }
    }

    fn draw(&mut self, terminal: &mut Terminal<impl Backend>) -> anyhow::Result<()> {
        terminal.draw(|frame| {
            self.custom_widget.render(
                frame.area(),
                frame.buffer_mut(),
                &self.custom_widget_data_state,
            );
        })?;
        Ok(())
    }
}
