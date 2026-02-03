# TermShell

A feature-rich terminal UI dashboard CLI tool – like `htop`, `btop`, and a file browser combined.

## Installation

```bash
git clone <repo-url>
cd terminal_ui
cargo install --path .
```

## Usage

Run from anywhere:
```bash
terminal_ui
```

### Command-line options
- `--dir <DIR>` / `-d <DIR>` – Start in a specific directory
- `--config <FILE>` / `-c <FILE>` – Load configuration from file

## Features

### 🎯 Interactive Dashboard
- **Real-time system metrics** – CPU, Memory, Disk usage gauges with color coding
- **Process monitoring** – Live process list with CPU/Memory usage
- **Network monitoring** – Network interface stats (sent/received bytes, packets)
- **File browser** – Navigate directories with icons, enter folders, go back with `h`
- **System logs** – Scrollable log viewer
- **Search** – Global search across files and processes (`/` to open)
- **Settings** – Interactive configuration with theme switching, refresh rate control

### ⌨️ Controls
- `h`/`l` or `←`/`→` – Switch tabs (or adjust settings in Settings tab)
- `j`/`k` or `↑`/`↓` – Navigate items within tabs
- `/` – Open search tab
- `Enter` – Select item / enter directory / save configuration
- `h` (in Files tab) – Go to parent directory
- `l` (in Files tab) – Enter selected directory
- `Space` – Toggle settings (show hidden files, auto refresh)
- `Backspace` – Delete character in search
- `q`/`Esc` – Exit
- `Tab` / `Shift+Tab` – Quick tab navigation

### 🎨 Visual Features
- **Colored gauges** for system resources
- **Highlighted selection** with inverse colors
- **Icons** for files/directories (📁/📄)
- **Responsive layout** adapts to terminal size
- **Tab indicator** showing active module
- **Footer bar** with keyboard shortcuts

### ⚙️ Configuration
Settings are saved to `terminal_ui_config.json` on exit:
- Refresh rate (ms) – Adjustable with arrow keys
- Theme preference – Switch between dark, light, blue themes
- Show hidden files option – Toggle with Space key
- Auto refresh toggle – Enable/disable automatic updates

## Requirements
- Rust 1.92+
- Interactive terminal (TTY)
- Linux/macOS/Windows with system info APIs

## Build from Source

```bash
cargo build --release
./target/release/terminal_ui
```

## Dependencies
- `ratatui` – Terminal UI framework
- `crossterm` – Cross-platform terminal handling
- `sysinfo` – System information
- `clap` – CLI argument parsing
- `serde` – Configuration serialization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License – see LICENSE file for details