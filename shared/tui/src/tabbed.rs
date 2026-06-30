use crate::CustomWidget;
use crossterm::event::{Event, KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Tabs, Widget},
};

pub struct TabbedWidget<T: CustomWidgetTuple> {
    widgets: T,
    current_tab: usize,
    tab_titles: Vec<String>,
}

pub trait CustomWidgetTuple: Send + 'static {
    type Data: Default + Send + 'static;
    fn len(&self) -> usize;
    fn render_at(&mut self, index: usize, area: Rect, buf: &mut Buffer, state: &Self::Data);
    fn on_ui_event_at(&mut self, index: usize, event: &Event);
}

impl<T: CustomWidgetTuple> TabbedWidget<T> {
    pub fn new<S: ToString>(widgets: T, tab_titles: &[S]) -> Self {
        Self {
            widgets,
            current_tab: 0,
            tab_titles: tab_titles.iter().map(|x| x.to_string()).collect(),
        }
    }

    fn get_tab_from_key(&self, code: &KeyCode) -> Option<usize> {
        match code {
            KeyCode::Char(c) => c.to_digit(10).and_then(|d| (d as usize).checked_sub(1)),
            _ => None,
        }
    }

    fn render_tab_bar(&self, area: Rect, buf: &mut Buffer) {
        let tabs = Tabs::new(
            self.tab_titles
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    Span::styled(
                        format!("[{}] {t}", i + 1),
                        Style::default().fg(Color::White),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .select(self.current_tab)
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .divider("|");

        let block = Block::default().borders(Borders::BOTTOM);
        tabs.block(block).render(area, buf);
    }
}

impl<T: CustomWidgetTuple> CustomWidget for TabbedWidget<T> {
    type Data = T::Data;

    fn render(&mut self, area: Rect, buf: &mut Buffer, state: &Self::Data) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(0)].as_ref())
            .split(area);

        self.render_tab_bar(chunks[0], buf);
        self.widgets
            .render_at(self.current_tab, chunks[1], buf, state);
    }

    fn on_ui_event(&mut self, event: &Event) {
        if let Event::Key(KeyEvent { code, .. }) = event {
            if let Some(new_tab) = self.get_tab_from_key(code) {
                if new_tab < self.widgets.len() {
                    self.current_tab = new_tab;
                    return;
                }
            }
        }
        self.widgets.on_ui_event_at(self.current_tab, event);
    }
}

macro_rules! tuple_len {
    ($($type:ident),+) => {
        [$(tuple_len!(@unit $type)),+].len()
    };
    (@unit $type:ident) => {
        ()
    };
}

macro_rules! impl_custom_widget_tuple {
    ($(($type:ident, $index:tt)),+) => {
        impl<$($type),+> CustomWidgetTuple for ($($type,)+)
        where
            $($type: CustomWidget),+
        {
            type Data = ($($type::Data,)+);

            fn len(&self) -> usize {
                tuple_len!($($type),+)
            }

            fn render_at(&mut self, index: usize, area: Rect, buf: &mut Buffer, state: &Self::Data) {
                match index {
                    $($index => self.$index.render(area, buf, &state.$index),)+
                    _ => {}
                }
            }

            fn on_ui_event_at(&mut self, index: usize, event: &Event) {
                match index {
                    $($index => self.$index.on_ui_event(event),)+
                    _ => {}
                }
            }
        }
    };
}

impl_custom_widget_tuple!((T1, 0));
impl_custom_widget_tuple!((T1, 0), (T2, 1));
impl_custom_widget_tuple!((T1, 0), (T2, 1), (T3, 2));
impl_custom_widget_tuple!((T1, 0), (T2, 1), (T3, 2), (T4, 3));
impl_custom_widget_tuple!((T1, 0), (T2, 1), (T3, 2), (T4, 3), (T5, 4));
