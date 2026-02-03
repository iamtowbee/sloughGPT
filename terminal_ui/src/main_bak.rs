use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

#[cfg(target_os = "macos")]
use nix;
#[cfg(target_os = "linux")]
use procfs;
#[cfg(target_os = "windows")]
use winapi;

use clap::Command as ClapCommand;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph};
use serde::{Deserialize, Serialize};
use sysinfo::System;

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

#[derive(Debug, Serialize, Deserialize, Default)]
struct Config {
    refresh_rate: u64,
    theme: String,
    show_hidden: bool,
    auto_refresh: bool,
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
    last_tick: std::time::Instant,
    config: Config,
    system: System,
    current_dir: std::path::PathBuf,

    // Desktop environment monitoring
    #[cfg(target_os = "macos")]
    macos_services: std::collections::HashMap<String, String>,
    #[cfg(target_os = "linux")]
    linux_services: std::collections::HashMap<String, String>,
    #[cfg(target_os = "windows")]
    windows_services: std::collections::HashMap<String, String>,
}

    // Desktop environment monitoring methods
    #[cfg(target_os = "macos")]
    fn get_macos_services(&mut self) -> std::io::Result<(), Box<dyn std::error::Error>> {
        use nix::unistd::{getuid, getgid};
        use nix::sys::{sysctl, uname, sysinfo, pid, stats, psinfo};
        
        let services = vec![
            ("launchd".to_string(), "System launch daemon".to_string()),
            ("windowserver".to_string(), "Window server".to_string()),
            ("coreaudiod".to_string(), "Core Audio daemon".to_string()),
            ("bluetoothd".to_string(), "Bluetooth daemon".to_string()),
            ("mDNSResponder".to_string(), "mDNS responder".to_string()),
            ("cfprefsd".to_string(), "Core Preferences daemon".to_string()),
            ("airportd".to_string(), "Airport daemon".to_string()),
            ("cupsd".to_string(), "Printing daemon".to_string()),
            ("pbs".to_string(), "Pasteboard services".to_string()),
            ("loginwindow".to_string(), "Login window agent".to_string()),
        ];
        
        for (name, description) in services {
            self.macos_services.insert(name.to_string(), description.to_string());
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn get_linux_services(&mut self) -> std::io::Result<(), Box<dyn std::error::Error>> {
        use procfs::{process::ProcessExt, process::CmdLine};
        use std::process::Command;
        
        // Get systemd services
        let output = Command::new("systemctl")
            .args(&["list-units", "--type=service", "--state=active"])
            .output()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            
        let lines = std::str::from_utf8(&output.stdout).lines();
        for line in lines {
            if line.contains("systemd-journald.service") && line.contains("active") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let service_name = parts[0].replace(".service", "");
                    let status = if parts.len() > 3 { "running".to_string() } else { "active".to_string() };
                    self.linux_services.insert(service_name.to_string(), status.to_string());
                }
            }
        }
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn get_windows_services(&mut self) -> std::io::Result<(), Box<dyn std::error::Error>> {
        use winapi::um::{LMSSTR, QueryServiceConfig, SC_HANDLE, SERVICE_STATUS_PROCESS_INFO};
        use std::ffi::c_void;
        use std::ptr;
        
        let services = vec![
            ("Spooler".to_string(), "Print Spooler".to_string()),
            ("EventLog".to_string(), "Event Log".to_string()),
            ("BITS".to_string(), "Background Intelligent Transfer".to_string()),
            ("Dhcp".to_string(), "DHCP Client".to_string()),
            ("Dnscache".to_string(), "DNS Cache".to_string()),
            ("LanmanServer".to_string(), "LAN Manager Server".to_string()),
            ("Browser".to_string(), "Browser Helper".to_string()),
        ];
        
        for (name, description) in services {
            self.windows_services.insert(name.to_string(), description.to_string());
        }
        Ok(())
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
            Tab::Files => self.selected_file = (self.selected_file + 1) % 4,
            Tab::Processes => self.selected_process = (self.selected_process + 1) % 4,
            Tab::Network => {
                self.selected_network_interface = (self.selected_network_interface + 1) % 3
            }
            Tab::Logs => self.selected_log = (self.selected_log + 1) % 5,
            Tab::Settings => self.selected_setting = (self.selected_setting + 1) % 4,
            Tab::Search => self.selected_file = (self.selected_file + 1) % 4,
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
                self.selected_file = if self.selected_file == 0 {
                    3
                } else {
                    self.selected_file - 1
                }
            }
            Tab::Processes => {
                self.selected_process = if self.selected_process == 0 {
                    3
                } else {
                    self.selected_process - 1
                }
            }
            Tab::Network => {
                self.selected_network_interface = if self.selected_network_interface == 0 {
                    2
                } else {
                    self.selected_network_interface - 1
                }
            }
            Tab::Logs => {
                self.selected_log = if self.selected_log == 0 {
                    4
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
                self.selected_file = if self.selected_file == 0 {
                    3
                } else {
                    self.selected_file - 1
                }
            }
        }
    }

    fn tick(&mut self) {
        let elapsed = self.last_tick.elapsed();
        if elapsed > Duration::from_millis(self.config.refresh_rate) {
            self.system.refresh_all();
            self.last_tick = std::time::Instant::now();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let matches = ClapCommand::new("terminal_ui")
        .version("0.1.0")
        .about("A terminal UI dashboard CLI tool")
        .get_matches();

    let mut app = App::new()?;

    // Terminal initialization
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    loop {
        app.tick();
        terminal.draw(|f| ui(f, &mut app))?;

        if crossterm::event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('h') | KeyCode::Left => app.prev_tab(),
                    KeyCode::Char('l') | KeyCode::Right => app.next_tab(),
                    KeyCode::Char('j') | KeyCode::Down => app.next_item(),
                    KeyCode::Char('k') | KeyCode::Up => app.prev_item(),
                    _ => {}
                },
                _ => {}
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}

fn ui(frame: &mut ratatui::Frame, app: &mut App) {
    let full_screen = frame.area();

    // Render main content taking full screen
    render_main_content(frame, full_screen, app);

    // Render overlay elements
    render_header_overlay(frame, full_screen, app);
    render_footer_overlay(frame, full_screen, app);
}

fn render_header_overlay(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let header_area = ratatui::layout::Rect {
        x: 0,
        y: 0,
        width: area.width,
        height: 3,
    };

    let tab_names = [
        "Dashboard",
        "Files",
        "Processes",
        "Network",
        "Logs",
        "Settings",
        "Search",
    ];

    let tabs: Vec<Span> = tab_names
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let tab_type = match i {
                0 => Tab::Dashboard,
                1 => Tab::Files,
                2 => Tab::Processes,
                3 => Tab::Network,
                4 => Tab::Logs,
                5 => Tab::Settings,
                6 => Tab::Search,
                _ => Tab::Dashboard,
            };
            create_tab_span(name, tab_type, &app)
        })
        .collect();

    let tabs_paragraph = Paragraph::new(Line::from(tabs))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL & !Borders::BOTTOM)
                .title("TermShell"),
        );
    frame.render_widget(tabs_paragraph, header_area);
}

fn create_tab_span(text: &'static str, tab: Tab, current_state: &App) -> Span<'static> {
    if current_state.current_tab == tab {
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

fn render_main_content(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &mut App) {
    let content_area = ratatui::layout::Rect {
        x: 0,
        y: 3,
        width: area.width,
        height: area.height.saturating_sub(4),
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
}

fn render_dashboard(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
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

    let disk_usage = 78;
    let disk_label = format!("Disk: {}%", disk_usage);
    let disk_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Disk Usage"))
        .gauge_style(
            Style::default()
                .fg(Color::Red)
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
        Line::from("System Status: Running"),
        Line::from("CPU: "),
        Line::from("Memory: "),
        Line::from("Processes: "),
    ])
    .block(Block::default().borders(Borders::ALL).title("System Info"))
    .wrap(ratatui::widgets::Wrap { trim: true });
    frame.render_widget(info, chunks[3]);
}

fn render_files(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &mut App) {
    let items: Vec<ListItem> = vec![
        ListItem::new("📁 Documents"),
        ListItem::new("📁 Downloads"),
        ListItem::new("📁 Pictures"),
        ListItem::new("📁 config.txt"),
        ListItem::new("📄 main.rs"),
    ];

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Files"));
    frame.render_widget(list, area);
}

fn render_processes(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let items: Vec<ListItem> = vec![
        ListItem::new("chrome: 12.5% CPU  512MB RAM"),
        ListItem::new("firefox: 8.3% CPU  256MB RAM"),
        ListItem::new("terminal: 2.1% CPU  64MB RAM"),
        ListItem::new("vscode: 5.7% CPU  128MB RAM"),
    ];

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Processes"));
    frame.render_widget(list, area);
}

fn render_network(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let items: Vec<ListItem> = vec![
        ListItem::new("eth0: 192.168.1.100 - Up/Down: 2.1MB/1.5MB"),
        ListItem::new("wlan0: 10.0.1.5 - Up/Down: 15.2MB/8.7MB"),
        ListItem::new("lo: 127.0.0.1 - Up/Down: 512KB/512KB"),
    ];

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Network"));
    frame.render_widget(list, area);
}

fn render_logs(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let items: Vec<ListItem> = vec![
        ListItem::new("INFO: System initialized"),
        ListItem::new("INFO: Loading dashboard modules"),
        ListItem::new("WARN: High memory usage detected"),
        ListItem::new("INFO: Memory usage normalized"),
    ];

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("System Logs"));
    frame.render_widget(list, area);
}

fn render_settings(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let items: Vec<ListItem> = vec![
        ListItem::new(format!("Refresh Rate: {} ms", app.config.refresh_rate)),
        ListItem::new(format!("Theme: {}", app.config.theme)),
        ListItem::new(format!("Show Hidden Files: {}", app.config.show_hidden)),
    ];

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Settings"));
    frame.render_widget(list, area);
}

fn render_search(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, _app: &App) {
    let search_text =
        Paragraph::new("Search: _").block(Block::default().borders(Borders::ALL).title("Search"));
    frame.render_widget(search_text, area);
}

fn render_footer_overlay(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let footer_area = ratatui::layout::Rect {
        x: 0,
        y: area.height.saturating_sub(1),
        width: area.width,
        height: 1,
    };

    let footer = Paragraph::new(Span::styled(
        "h/l: tabs | j/k: navigate | q: quit",
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    ))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL & !Borders::TOP));
    frame.render_widget(footer, footer_area);
}
