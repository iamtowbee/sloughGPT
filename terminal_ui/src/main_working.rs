use std::error::Error;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Arg, Command as ClapCommand};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseEventKind,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, Gauge, List, ListItem, ListState, Paragraph, Row, Table, Wrap,
};
use serde::{Deserialize, Serialize};
use signal_hook::{consts::SIGINT, iterator::Signals};
use sysinfo::{Pid, Process, System};

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Dashboard,
    Files,
    Processes,
    Network,
    Logs,
    Settings,
    Search,
}

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    refresh_rate: u64,
    theme: String,
    show_hidden: bool,
    auto_refresh: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            refresh_rate: 2000,
            theme: "dark".to_string(),
            show_hidden: false,
            auto_refresh: true,
        }
    }
}

struct App {
    tabs: Vec<Tab>,
    current_tab: Tab,
    selected_dashboard_item: usize,
    selected_file: usize,
    selected_process: usize,
    selected_network_interface: usize,
    selected_log: usize,
    selected_setting: usize,
    state: ListState,
    last_tick: Instant,
    config: Config,
    system: System,
    search_query: String,
    search_results: Vec<String>,
    current_dir: PathBuf,
    logs: Vec<String>,
    processes: Vec<ProcessInfo>,
    files: Vec<FileInfo>,
    network_interfaces: Vec<NetworkInterface>,
    file_preview: Option<FilePreview>,
    show_preview: bool,
}

#[derive(Clone, Debug)]
struct FilePreview {
    path: PathBuf,
    content: Vec<String>,
    line_count: usize,
    file_type: PreviewType,
    scroll_offset: usize,
}

#[derive(Clone, Debug)]
enum PreviewType {
    Text,
    Binary,
    Image,
    Directory,
    Error(String),
}

#[derive(Clone, Debug)]
struct FileInfo {
    name: String,
    path: PathBuf,
    size: u64,
    is_dir: bool,
    modified: String,
}

impl FileInfo {
    fn new(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        let metadata = std::fs::metadata(&path)?;
        let name = path
            .file_name()
            .ok_or_else(|| "Invalid filename")?
            .to_string_lossy()
            .to_string();
        Ok(Self {
            name,
            path: path.clone(),
            size: metadata.len(),
            is_dir: metadata.is_dir(),
            modified: format!("{:?}", metadata.modified()?),
        })
    }
}

#[derive(Clone, Debug)]
struct ProcessInfo {
    pid: Pid,
    name: String,
    cpu_usage: f32,
    memory_usage: u64,
    status: String,
}

#[derive(Clone, Debug)]
struct NetworkInterface {
    name: String,
    ip: Option<String>,
    bytes_sent: u64,
    bytes_received: u64,
    packets_sent: u64,
    packets_received: u64,
    status: String,
}

impl From<&Process> for ProcessInfo {
    fn from(process: &Process) -> Self {
        Self {
            pid: process.pid(),
            name: process.name().to_string(),
            cpu_usage: process.cpu_usage(),
            memory_usage: process.memory(),
            status: format!("{:?}", process.status()),
        }
    }
}

impl App {
    fn new() -> Result<Self, Box<dyn Error>> {
        let tabs = vec![
            Tab::Dashboard,
            Tab::Files,
            Tab::Processes,
            Tab::Network,
            Tab::Logs,
            Tab::Settings,
            Tab::Search,
        ];
        let mut state = ListState::default();
        state.select(Some(0));

        let current_dir = std::env::current_dir()?;

        Ok(Self {
            tabs,
            current_tab: Tab::Dashboard,
            selected_dashboard_item: 0,
            selected_file: 0,
            selected_process: 0,
            selected_network_interface: 0,
            selected_log: 0,
            selected_setting: 0,
            state,
            last_tick: Instant::now(),
            config: Config::default(),
            system: System::new(),
            search_query: String::new(),
            search_results: Vec::new(),
            current_dir,
            logs: vec![
                "2024-01-28 10:23:45 [INFO] System initialized".to_string(),
                "2024-01-28 10:23:46 [INFO] Loading dashboard modules".to_string(),
                "2024-01-28 10:23:47 [INFO] File system watcher started".to_string(),
                "2024-01-28 10:23:48 [WARN] High memory usage detected".to_string(),
                "2024-01-28 10:23:49 [INFO] Memory usage normalized".to_string(),
            ],
            processes: Vec::new(),
            files: Vec::new(),
            network_interfaces: vec![
                NetworkInterface {
                    name: "eth0".to_string(),
                    ip: Some("192.168.1.100".to_string()),
                    bytes_sent: 1024000,
                    bytes_received: 2048000,
                    packets_sent: 1500,
                    packets_received: 2100,
                    status: "UP".to_string(),
                },
                NetworkInterface {
                    name: "lo".to_string(),
                    ip: Some("127.0.0.1".to_string()),
                    bytes_sent: 512000,
                    bytes_received: 512000,
                    packets_sent: 800,
                    packets_received: 800,
                    status: "UP".to_string(),
                },
            ],
            file_preview: None,
            show_preview: false,
        })
    }

    fn next_tab(&mut self) {
        let current_idx = self
            .tabs
            .iter()
            .position(|&t| t == self.current_tab)
            .unwrap();
        let next_idx = (current_idx + 1) % self.tabs.len();
        self.current_tab = self.tabs[next_idx];
    }

    fn prev_tab(&mut self) {
        let current_idx = self
            .tabs
            .iter()
            .position(|&t| t == self.current_tab)
            .unwrap();
        let prev_idx = if current_idx == 0 {
            self.tabs.len() - 1
        } else {
            current_idx - 1
        };
        self.current_tab = self.tabs[prev_idx];
    }

    fn next_item(&mut self) {
        match self.current_tab {
            Tab::Dashboard => self.selected_dashboard_item = (self.selected_dashboard_item + 1) % 6,
            Tab::Files => {
                self.selected_file = if self.files.is_empty() {
                    0
                } else {
                    (self.selected_file + 1) % self.files.len()
                };
                if self.show_preview {
                    self.load_file_preview();
                }
            }
            Tab::Processes => {
                self.selected_process = if self.processes.is_empty() {
                    0
                } else {
                    (self.selected_process + 1) % self.processes.len()
                }
            }
            Tab::Network => {
                self.selected_network_interface = if self.network_interfaces.is_empty() {
                    0
                } else {
                    (self.selected_network_interface + 1) % self.network_interfaces.len()
                }
            }
            Tab::Logs => {
                self.selected_log = if self.logs.is_empty() {
                    0
                } else {
                    (self.selected_log + 1) % self.logs.len()
                }
            }
            Tab::Settings => self.selected_setting = (self.selected_setting + 1) % 4,
            Tab::Search => {
                self.selected_file = if self.search_results.is_empty() {
                    0
                } else {
                    (self.selected_file + 1) % self.search_results.len()
                }
            }
        }
    }

fn prev_item(&mut self) {
        match self.current_tab {
            Tab::Dashboard => {
                self.selected_dashboard_item = if self.selected_dashboard_item == 0 {
                    5
                } else {
                    self.selected_dashboard_item - 1
                }
            }
            Tab::Files => {
                self.selected_file = if self.files.is_empty() || self.selected_file == 0 {
                    self.files.len().saturating_sub(1)
                } else {
                    self.selected_file - 1
                };
                if self.show_preview {
                    self.load_file_preview();
                }
            }
            Tab::Processes => {
                self.selected_process = if self.processes.is_empty() || self.selected_process == 0 {
                    self.processes.len().saturating_sub(1)
                } else {
                    self.selected_process - 1
                }
            }
            Tab::Network => {
                self.selected_network_interface = if self.network_interfaces.is_empty() || self.selected_network_interface == 0 {
                    self.network_interfaces.len().saturating_sub(1)
                } else {
                    self.selected_network_interface - 1
                }
            }
            Tab::Logs => {
                self.selected_log = if self.logs.is_empty() || self.selected_log == 0 {
                    self.logs.len().saturating_sub(1)
                } else {
                    self.selected_log - 1
                }
            }
            Tab::Settings => {
                self.selected_setting = if self.selected_setting == 0 {
                    3
                } else {
                    self.selected_setting - 1
                }
            }
            Tab::Search => {
                self.selected_file = if self.search_results.is_empty() || self.selected_file == 0 {
                    self.search_results.len().saturating_sub(1)
                } else {
                    self.selected_file - 1
                }
            }
        }
    }

    fn update_files(&mut self) -> Result<(), Box<dyn Error>> {
        self.files.clear();
        let mut entries = std::fs::read_dir(&self.current_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                if self.config.show_hidden {
                    true
                } else {
                    !entry.file_name().to_string_lossy().starts_with('.')
                }
            })
            .filter_map(|entry| FileInfo::new(entry.path()).ok())
            .collect::<Vec<_>>();

        entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        });

        self.files = entries;
        Ok(())
    }

    fn update_processes(&mut self) {
        self.system.refresh_all();
        self.processes.clear();

        for process in self.system.processes().values() {
            self.processes.push(ProcessInfo::from(process));
        }

        self.processes
            .sort_by(|a, b| b.cpu_usage.partial_cmp(&a.cpu_usage).unwrap());
    }

    fn navigate_up(&mut self) {
        if let Some(parent) = self.current_dir.parent() {
            self.current_dir = parent.to_path_buf();
            let _ = self.update_files();
        }
    }

    fn navigate_into(&mut self) {
        if let Some(selected) = self.files.get(self.selected_file) {
            if selected.is_dir {
                self.current_dir = selected.path.clone();
                let _ = self.update_files();
            }
        }
    }

    fn search(&mut self, query: &str) {
        self.search_query = query.to_string();
        self.search_results.clear();

        if query.is_empty() {
            return;
        }

        let query_lower = query.to_lowercase();

        for file in &self.files {
            if file.name.to_lowercase().contains(&query_lower) {
                self.search_results.push(file.name.clone());
            }
        }

        for process in &self.processes {
            if process.name.to_lowercase().contains(&query_lower) {
                self.search_results
                    .push(format!("Process: {}", process.name));
            }
        }
    }

    fn tick(&mut self) {
        let elapsed = self.last_tick.elapsed();
        if elapsed > Duration::from_millis(self.config.refresh_rate) {
            self.update_processes();
            self.last_tick = Instant::now();
        }
    }

    fn toggle_preview(&mut self) {
        self.show_preview = !self.show_preview;
        if self.show_preview {
            self.load_file_preview();
        } else {
            self.file_preview = None;
        }
    }

    fn load_file_preview(&mut self) {
        if let Some(selected_file) = self.files.get(self.selected_file) {
            if selected_file.is_dir {
                self.file_preview = Some(FilePreview {
                    path: selected_file.path.clone(),
                    content: vec!["Directory preview not available".to_string()],
                    line_count: 1,
                    file_type: PreviewType::Directory,
                    scroll_offset: 0,
                });
                return;
            }

            match self.read_file_content(&selected_file.path) {
                Ok((content, file_type)) => {
                    self.file_preview = Some(FilePreview {
                        path: selected_file.path.clone(),
                        content,
                        line_count: content.len(),
                        file_type,
                        scroll_offset: 0,
                    });
                }
                Err(e) => {
                    self.file_preview = Some(FilePreview {
                        path: selected_file.path.clone(),
                        content: vec![format!("Error reading file: {}", e)],
                        line_count: 1,
                        file_type: PreviewType::Error(e.to_string()),
                        scroll_offset: 0,
                    });
                }
            }
        }
    }

    fn read_file_content(
        &self,
        path: &PathBuf,
    ) -> Result<(Vec<String>, PreviewType), Box<dyn Error>> {
        const MAX_PREVIEW_SIZE: usize = 1024 * 10; // 10KB limit for preview

        let metadata = std::fs::metadata(path)?;
        if metadata.len() > MAX_PREVIEW_SIZE as u64 {
            return Ok((
                vec!["File too large for preview (>10KB)".to_string()],
                PreviewType::Text,
            ));
        }

        // Try to detect file type by extension
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        let is_text = match extension.to_lowercase().as_str() {
            "txt" | "rs" | "py" | "js" | "html" | "css" | "json" | "md" | "toml" | "yaml"
            | "yml" | "xml" | "sh" | "bat" | "log" => true,
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" => false,
            "pdf" | "zip" | "tar" | "gz" | "exe" | "bin" => false,
            _ => {
                // Try to read first few bytes to detect if it's text
                let mut file = std::fs::File::open(path)?;
                let mut buffer = [0; 512];
                let bytes_read = file.read(&mut buffer)?;
                buffer[..bytes_read]
                    .iter()
                    .all(|&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
            }
        };

        if !is_text {
            return Ok((
                vec![format!("Binary file: {}", extension)],
                PreviewType::Binary,
            ));
        }

        let content = std::fs::read_to_string(path)?;
        let lines: Vec<String> = content.lines().map(|line| line.to_string()).collect();
        Ok((lines, PreviewType::Text))
    }

    fn scroll_preview(&mut self, direction: isize) {
        if let Some(ref mut preview) = self.file_preview {
            let max_scroll = preview.line_count.saturating_sub(1);
            match direction {
                1 => {
                    preview.scroll_offset = preview.scroll_offset.saturating_add(1).min(max_scroll)
                }
                -1 => preview.scroll_offset = preview.scroll_offset.saturating_sub(1),
                _ => {}
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let matches = ClapCommand::new("terminal_ui")
        .version("0.1.0")
        .about("A terminal UI dashboard CLI tool")
        .arg(
            Arg::new("directory")
                .short('d')
                .long("dir")
                .value_name("DIR")
                .help("Start in specific directory"),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Path to config file"),
        )
        .arg(
            Arg::new("search")
                .long("search")
                .value_name("QUERY")
                .help("Run search in separate process"),
        )
        .get_matches();

    // Handle search mode (separate process)
    if let Some(query) = matches.get_one::<String>("search") {
        println!("Searching for: {}", query);
        let mut app = App::new()?;
        app.current_tab = Tab::Search;
        app.search_query = query.clone();
        app.search(&query);

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        crossterm::execute!(stdout, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        loop {
            terminal.draw(|f| {
                let search_area = f.area();
                render_search(f, search_area, &app);
            })?;

            if crossterm::event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                        break;
                    }
                }
            }
        }

        disable_raw_mode()?;
        crossterm::execute!(terminal.backend_mut(), DisableMouseCapture)?;
        terminal.show_cursor()?;
        return Ok(());
    }

    let mut app = App::new()?;

    if let Some(dir) = matches.get_one::<String>("directory") {
        app.current_dir = PathBuf::from(dir);
        app.update_files()?;
    }

    if let Some(config_file) = matches.get_one::<String>("config") {
        if let Ok(config_str) = std::fs::read_to_string(config_file) {
            if let Ok(config) = serde_json::from_str::<Config>(&config_str) {
                app.config = config;
            }
        }
    }

    app.update_files()?;
    app.update_processes();

    // Terminal initialization
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Setup signal handling for Ctrl+C
    let mut signals = Signals::new(&[SIGINT])?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for signal in signals.forever() {
            let _ = tx.send(signal);
        }
    });

    loop {
        app.tick();
        terminal.draw(|f| ui(f, &mut app))?;

        // Check for Ctrl+C signal
        if rx.try_recv().is_ok() {
            break;
        }

        if crossterm::event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
KeyCode::Char('h') | KeyCode::Left => {
                        if app.current_tab == Tab::Files && app.show_preview {
                            app.scroll_preview(-1);
                        } else if app.current_tab == Tab::Settings {
                            match app.selected_setting {
                                0 => app.config.refresh_rate = app.config.refresh_rate.saturating_sub(500),
                                1 => {
                                    let themes = vec!["dark", "light", "blue"];
                                    let current_idx = themes.iter().position(|&t| t == app.config.theme).unwrap_or(0);
                                    let prev_idx = if current_idx == 0 { themes.len() - 1 } else { current_idx - 1 };
                                    app.config.theme = themes[prev_idx].to_string();
                                }
                                _ => {}
                            }
                        } else {
                            app.prev_tab();
                        }
                    },
                    KeyCode::Char('l') | KeyCode::Right => {
                        if app.current_tab == Tab::Files && app.show_preview {
                            app.scroll_preview(1);
                        } else if app.current_tab == Tab::Settings {
                            match app.selected_setting {
                                0 => app.config.refresh_rate = app.config.refresh_rate.saturating_add(500),
                                1 => {
                                    let themes = vec!["dark", "light", "blue"];
                                    let current_idx = themes.iter().position(|&t| t == app.config.theme).unwrap_or(0);
                                    let next_idx = (current_idx + 1) % themes.len();
                                    app.config.theme = themes[next_idx].to_string();
                                }
                                _ => {}
                            }
                        } else {
                            app.next_tab();
                        }
                    },
                Event::Mouse(event) => {
                    match event.kind {
                        MouseEventKind::Down(_) => {
                            // Handle mouse clicks
                        }
                        MouseEventKind::ScrollUp => app.prev_item(),
                        MouseEventKind::ScrollDown => app.next_item(),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    // Save config on exit
    if let Ok(config_str) = serde_json::to_string_pretty(&app.config) {
        let _ = std::fs::write("terminal_ui_config.json", config_str);
    }

    // Restore terminal
    disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}

fn ui(frame: &mut Frame, app: &mut App) {
    let full_screen = frame.area();

    // Render main content taking full screen
    render_main_content(frame, full_screen, app);

    // Render overlay elements
    render_header_overlay(frame, full_screen, app);
    render_footer_overlay(frame, full_screen);
}

fn render_header_overlay(frame: &mut Frame, area: Rect, app: &App) {
    let header_area = Rect {
        x: 0,
        y: 0,
        width: area.width,
        height: 3,
    };

    // Tabs in header
    let tab_names = [
        "Dashboard",
        "Files",
        "Processes",
        "Network",
        "Logs",
        "Settings",
        "Search",
    ];
    let tabs = vec![
        Span::raw(" "),
        create_tab_span(tab_names[0], Tab::Dashboard, app),
        Span::raw(" "),
        create_tab_span(tab_names[1], Tab::Files, app),
        Span::raw(" "),
        create_tab_span(tab_names[2], Tab::Processes, app),
        Span::raw(" "),
        create_tab_span(tab_names[3], Tab::Network, app),
        Span::raw(" "),
        create_tab_span(tab_names[4], Tab::Logs, app),
        Span::raw(" "),
        create_tab_span(tab_names[5], Tab::Settings, app),
        Span::raw(" "),
        create_tab_span(tab_names[6], Tab::Search, app),
        Span::raw(" "),
    ];

    let tabs_paragraph = Paragraph::new(Line::from(tabs))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL & !Borders::BOTTOM)
                .title("TermShell"),
        );
    frame.render_widget(tabs_paragraph, header_area);

    // Title on the left side of header
    let title_area = Rect {
        x: 2,
        y: 1,
        width: 20,
        height: 1,
    };
    let title = Paragraph::new("v0.1.0")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM));
    frame.render_widget(title, title_area);
}

fn create_tab_span(text: &'static str, tab: Tab, app: &App) -> Span<'static> {
    if app.current_tab == tab {
        Span::styled(
            text,
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(text, Style::default().fg(Color::Gray))
    }
}

fn render_main_content(frame: &mut Frame, area: Rect, app: &mut App) {
    let content_area = Rect {
        x: 0,
        y: 3,
        width: area.width,
        height: area.height.saturating_sub(4), // Leave space for header and footer
    };

    match app.current_tab {
        Tab::Dashboard => render_dashboard(frame, content_area, app),
        Tab::Files => render_files(frame, content_area, app),
        Tab::Processes => render_processes(frame, content_area, app),
        Tab::Network => render_network(frame, content_area, app),
        Tab::Logs => render_logs(frame, content_area, app),
        Tab::Settings => render_settings(frame, content_area, app),
        Tab::Search => render_search(frame, content_area, app),
    }

    // System stats overlay in top-right corner (over header area)
    render_system_stats(frame, area, app);
}

fn render_dashboard(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

    let cpu_usage = app.system.global_cpu_info().cpu_usage() as u16;
    let total_memory = app.system.total_memory();
    let used_memory = app.system.used_memory();
    let memory_usage = (used_memory * 100 / total_memory) as u16;

    let cpu_label = format!("CPU: {}%", cpu_usage);
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("CPU Usage"))
        .gauge_style(
            Style::default()
                .fg(Color::Green)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .label(Span::styled(
            cpu_label,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ))
        .ratio(cpu_usage as f64 / 100.0);
    frame.render_widget(cpu_gauge, chunks[0]);

    let mem_label = format!("Memory: {}%", memory_usage);
    let mem_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Memory Usage"))
        .gauge_style(
            Style::default()
                .fg(Color::Yellow)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .label(Span::styled(
            mem_label,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ))
        .ratio(memory_usage as f64 / 100.0);
    frame.render_widget(mem_gauge, chunks[1]);

    let disk_usage = get_disk_usage(&app.current_dir);
    let disk_label = format!("Disk: {}%", disk_usage);
    let disk_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Disk Usage"))
        .gauge_style(
            Style::default()
                .fg(if disk_usage > 80 {
                    Color::Red
                } else if disk_usage > 50 {
                    Color::Yellow
                } else {
                    Color::Green
                })
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .label(Span::styled(
            disk_label,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ))
        .ratio(disk_usage as f64 / 100.0);
    frame.render_widget(disk_gauge, chunks[2]);

    let info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("System Status: ", Style::default().fg(Color::Cyan)),
            Span::styled("Running", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Processes: ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{}", app.processes.len()),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Uptime: ", Style::default().fg(Color::Cyan)),
            Span::styled("2h 34m 12s", Style::default().fg(Color::White)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title("System Info"))
    .wrap(Wrap { trim: true });
    frame.render_widget(info, chunks[3]);

    let help = Paragraph::new(vec![
        Line::from("Keyboard Shortcuts:"),
        Line::from("  h/l or ←/→: Switch tabs"),
        Line::from("  j/k or ↑/↓: Navigate items"),
        Line::from("  /: Search, Enter: Select, h: Parent dir"),
        Line::from("  q/Esc: Exit"),
    ])
    .block(Block::default().borders(Borders::ALL).title("Help"));
    frame.render_widget(help, chunks[4]);
}

fn render_files(frame: &mut Frame, area: Rect, app: &mut App) {
    let chunks = if app.show_preview {
        Layout::default()
            .direction(Direction::Horizontal)
            .margin(1)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area)
    };

if app.show_preview {
                            self.load_file_preview();
                        }
}

fn render_processes(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Min(0)])
        .split(area);

    let rows: Vec<Row> = app
        .processes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let style = if i == app.selected_process {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(p.pid.to_string()),
                Cell::from(p.name.clone()),
                Cell::from(format!("{:.1}%", p.cpu_usage)),
                Cell::from(format!("{} MB", p.memory_usage / 1024 / 1024)),
                Cell::from(p.status.clone()),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        &[
            Constraint::Length(8),
            Constraint::Min(15),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(12),
        ],
    )
    .header(Row::new(vec![
        Cell::from("PID").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Name").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("CPU %").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Memory").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Status").style(Style::default().add_modifier(Modifier::BOLD)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Running Processes"),
    );
    frame.render_widget(table, chunks[0]);
}

fn render_network(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Min(0)])
        .split(area);

    let rows: Vec<Row> = app
        .network_interfaces
        .iter()
        .enumerate()
        .map(|(i, iface)| {
            let style = if i == app.selected_network_interface {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(iface.name.clone()),
                Cell::from(iface.ip.clone().unwrap_or_else(|| "N/A".to_string())),
                Cell::from(format_bytes(iface.bytes_sent)),
                Cell::from(format_bytes(iface.bytes_received)),
                Cell::from(iface.packets_sent.to_string()),
                Cell::from(iface.packets_received.to_string()),
                Cell::from(iface.status.clone()),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        &[
            Constraint::Length(12),
            Constraint::Length(15),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
        ],
    )
    .header(Row::new(vec![
        Cell::from("Interface").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("IP Address").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Sent").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Received").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Pkt Sent").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Pkt Recv").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Status").style(Style::default().add_modifier(Modifier::BOLD)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Network Interfaces"),
    );
    frame.render_widget(table, chunks[0]);
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

fn get_disk_usage(current_dir: &PathBuf) -> u16 {
    match std::fs::metadata(current_dir) {
        Ok(_) => {
            // For simplicity, return a simulated disk usage percentage
            // In a real implementation, you'd use sysinfo or platform-specific APIs
            use std::time::SystemTime;
            let secs = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            (secs % 80 + 20) as u16 // Simulate between 20-99% usage
        }
        Err(_) => 0,
    }
}

fn render_logs(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .logs
        .iter()
        .enumerate()
        .map(|(i, log)| {
            let style = if i == app.selected_log {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(log.clone()).style(style)
        })
        .collect();

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("System Logs"));
    frame.render_widget(list, area);
}

fn render_settings(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    let title = Paragraph::new(Line::from(vec![Span::styled(
        "Settings",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )]))
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(title, chunks[0]);

    let settings = vec![
        format!("Refresh Rate: {} ms [↑/↓]", app.config.refresh_rate),
        format!("Theme: {} [←/→]", app.config.theme),
        format!(
            "Show Hidden Files: {} [Space]",
            if app.config.show_hidden { "Yes" } else { "No" }
        ),
        format!(
            "Auto Refresh: {} [Space]",
            if app.config.auto_refresh { "Yes" } else { "No" }
        ),
        "Save Configuration [Enter]".to_string(),
    ];

    let items: Vec<ListItem> = settings
        .iter()
        .enumerate()
        .map(|(i, setting)| {
            let style = if i == app.selected_setting {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(setting.clone()).style(style)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Configuration"),
    );
    frame.render_widget(list, chunks[1]);
}

fn render_search(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    let search_input = Paragraph::new(Line::from(vec![
        Span::styled("Search: ", Style::default().fg(Color::Cyan)),
        Span::styled(&app.search_query, Style::default().fg(Color::White)),
        Span::raw("_"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(search_input, chunks[0]);

    let items: Vec<ListItem> = app
        .search_results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let style = if i == app.selected_file {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(result.clone()).style(style)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Search Results"),
    );
    frame.render_widget(list, chunks[1]);
}

fn render_footer_overlay(frame: &mut Frame, area: Rect, app: &App) {
    let footer_text = if app.current_tab == Tab::Files {
        " h/l: tabs | j/k: navigate | Enter: dir | p: preview | ↑/↓: scroll | q: quit "
    } else {
        " h/l: tabs | j/k: navigate | /: search | Enter: select | h: parent dir | q: quit "
    };

    let footer_area = Rect {
        x: 0,
        y: area.height.saturating_sub(1),
        width: area.width,
        height: 1,
    };

    let footer = Paragraph::new(Span::styled(
        footer_text,
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    ))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL & !Borders::TOP));
    frame.render_widget(footer, footer_area);
}

fn render_file_preview(frame: &mut Frame, area: Rect, app: &mut App) {
    let preview_title = match app.file_preview.as_ref() {
        Some(preview) => match &preview.file_type {
            PreviewType::Text => "File Preview",
            PreviewType::Binary => "Binary File",
            PreviewType::Image => "Image File",
            PreviewType::Directory => "Directory",
            PreviewType::Error(_) => "Preview Error",
        },
        None => "No Preview",
    };

    let preview_content = match app.file_preview.as_ref() {
        Some(preview) => {
            let max_lines = (area.height.saturating_sub(2)) as usize;
            let start_line = preview.scroll_offset;
            let end_line = (start_line + max_lines).min(preview.content.len());

            preview.content[start_line..end_line]
                .iter()
                .enumerate()
                .map(|(i, line)| {
                    let line_num = start_line + i + 1;
                    let prefix = format!("{:>4}: ", line_num);
                    Line::from(vec![
                        Span::styled(prefix, Style::default().fg(Color::DarkGray)),
                        Span::styled(line.clone(), Style::default().fg(Color::White)),
                    ])
                })
                .collect::<Vec<_>>()
        }
        None => vec![Line::from("Select a file to preview")],
    };

    let preview_widget = Paragraph::new(preview_content)
        .block(Block::default().borders(Borders::ALL).title(preview_title))
        .wrap(Wrap { trim: true });

    frame.render_widget(preview_widget, area);
}

fn render_system_stats(frame: &mut Frame, area: Rect, app: &App) {
    let cpu_usage = app.system.global_cpu_info().cpu_usage() as u16;
    let total_memory = app.system.total_memory();
    let used_memory = app.system.used_memory();
    let memory_usage = (used_memory * 100 / total_memory) as u16;

    let stats_width = 25;
    let stats_height = 5;
    let stats_area = Rect {
        x: area.width.saturating_sub(stats_width + 1),
        y: 1,
        width: stats_width,
        height: stats_height,
    };

    let uptime_hours = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        / 3600.0
        % 24.0;

    let compact_stats = Paragraph::new(vec![
        Line::from(Span::styled(
            "SYSTEM",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("CPU:", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}%", cpu_usage),
                Style::default()
                    .fg(if cpu_usage > 80 {
                        Color::Red
                    } else if cpu_usage > 50 {
                        Color::Yellow
                    } else {
                        Color::Green
                    })
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("RAM:", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}%", memory_usage),
                Style::default()
                    .fg(if memory_usage > 80 {
                        Color::Red
                    } else if memory_usage > 50 {
                        Color::Yellow
                    } else {
                        Color::Green
                    })
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("PROCS:", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.processes.len()),
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("UP:", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}h", uptime_hours),
                Style::default().fg(Color::Magenta),
            ),
        ]),
    ])
    .alignment(Alignment::Left)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title("Stats"),
    );

    frame.render_widget(Clear, stats_area);
    frame.render_widget(compact_stats, stats_area);
}

