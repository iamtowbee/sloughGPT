"""
CLI Interface Implementation

This module provides command-line interface capabilities including
interactive commands, scripting support, and terminal output formatting.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ...__init__ import (
    BaseComponent,
    ComponentException,
    IUIController,
    ResponseFormat,
    UIRequest,
    UIResponse,
    UIType,
)


class CLIInterface(BaseComponent, IUIController):
    """Advanced CLI interface system"""

    def __init__(self) -> None:
        super().__init__("cli_interface")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # CLI state
        self.commands: Dict[str, Any] = {}
        self.command_history: List[str] = []
        self.current_session: Optional[Dict[str, Any]] = None

        # CLI configuration
        self.cli_config: Dict[str, Any] = {
            "prompt": "sloughgpt> ",
            "enable_colors": True,
            "enable_autocomplete": True,
            "history_size": 1000,
            "enable_help": True,
        }

        # CLI metrics
        self.cli_metrics = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "session_duration": 0.0,
        }

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize CLI interface"""
        try:
            self.logger.info("Initializing CLI Interface...")

            # Register default commands
            await self._register_default_commands()

            self.is_initialized = True
            self.logger.info("CLI Interface initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize CLI Interface: {e}")
            raise ComponentException(f"CLI Interface initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown CLI interface"""
        try:
            self.logger.info("Shutting down CLI Interface...")
            self.is_initialized = False
            self.logger.info("CLI Interface shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown CLI Interface: {e}")
            raise ComponentException(f"CLI Interface shutdown failed: {e}")

    async def handle_request(self, request: UIRequest) -> UIResponse:
        """Handle CLI request"""
        try:
            self.cli_metrics["total_commands"] += 1

            # Parse command
            command = request.parameters.get("command", "")
            args = request.parameters.get("args", [])

            # Execute command
            if command in self.commands:
                handler = self.commands[command]
                result = await handler(args)

                self.cli_metrics["successful_commands"] += 1

                return UIResponse(
                    request_id=request.request_id,
                    status="success",
                    data=result,
                    format=ResponseFormat.TEXT,
                    metadata={},
                    timestamp=time.time(),
                )
            else:
                self.cli_metrics["failed_commands"] += 1

                return UIResponse(
                    request_id=request.request_id,
                    status="error",
                    data=f"Unknown command: {command}",
                    format=ResponseFormat.TEXT,
                    metadata={},
                    timestamp=time.time(),
                )

        except Exception as e:
            self.logger.error(f"CLI request handling failed: {e}")
            self.cli_metrics["failed_commands"] += 1

            return UIResponse(
                request_id=request.request_id,
                status="error",
                data=str(e),
                format=ResponseFormat.TEXT,
                metadata={},
                timestamp=time.time(),
            )

    async def validate_request(self, request: UIRequest) -> bool:
        """Validate CLI request"""
        try:
            # Check if command is provided
            if "command" not in request.parameters:
                return False

            return True

        except Exception as e:
            self.logger.error(f"CLI request validation error: {e}")
            return False

    async def format_response(self, data: Any, format: ResponseFormat) -> Any:
        """Format CLI response"""
        try:
            if format == "text":
                if isinstance(data, str):
                    return data
                else:
                    return str(data)

            elif format == "json":
                return json.dumps(data, indent=2)

            elif format == "table":
                return self._format_as_table(data)

            else:
                return str(data)

        except Exception as e:
            self.logger.error(f"CLI response formatting error: {e}")
            return f"Error: {e}"

    async def start_interactive_mode(self) -> None:
        """Start interactive CLI mode"""
        try:
            self.logger.info("Starting interactive CLI mode...")

            # Start session
            self.current_session = {"start_time": time.time(), "commands_executed": 0}

            print("SloughGPT CLI Interface")
            print("Type 'help' for available commands")
            print("Type 'exit' to quit")
            print()

            # Main command loop
            while self.current_session:
                try:
                    # Get user input
                    command_input = input(self.cli_config["prompt"]).strip()

                    if not command_input:
                        continue

                    # Add to history
                    self.command_history.append(command_input)
                    if len(self.command_history) > self.cli_config["history_size"]:
                        self.command_history = self.command_history[
                            -self.cli_config["history_size"] :
                        ]

                    # Parse command
                    parts = command_input.split()
                    command = parts[0]
                    args = parts[1:] if len(parts) > 1 else []

                    # Handle special commands
                    if command == "exit":
                        break
                    elif command == "help":
                        await self._show_help()
                        continue

                    # Execute command
                    request = UIRequest(
                        request_id=f"cli_{int(time.time() * 1000)}",
                        ui_type=UIType.CLI,
                        endpoint="/execute",
                        parameters={"command": command, "args": args},
                        user=None,
                        timestamp=time.time(),
                    )

                    response = await self.handle_request(request)

                    # Display response
                    if response.status == "success":
                        formatted_output = await self.format_response(
                            response.data, ResponseFormat.TEXT
                        )
                        print(formatted_output)
                    else:
                        print(f"Error: {response.data}")

                    print()

                    self.current_session["commands_executed"] += 1

                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit")
                except EOFError:
                    break

            # End session
            if self.current_session:
                session_duration = time.time() - self.current_session["start_time"]
                self.cli_metrics["session_duration"] = session_duration
                self.current_session = None

            print("Goodbye!")

        except Exception as e:
            self.logger.error(f"Interactive CLI mode error: {e}")

    async def execute_command(self, command: str, args: Optional[List[str]] = None) -> str:
        """Execute a single CLI command"""
        try:
            if args is None:
                args = []

            request = UIRequest(
                request_id=f"cli_{int(time.time() * 1000)}",
                ui_type=UIType.CLI,
                endpoint="/execute",
                parameters={"command": command, "args": args},
                user=None,
                timestamp=time.time(),
            )

            response = await self.handle_request(request)

            if response.status == "success":
                result = await self.format_response(response.data, ResponseFormat.TEXT)
                return str(result) if result is not None else ""
            else:
                return f"Error: {response.data}"

        except Exception as e:
            return f"Command execution error: {e}"

    async def get_cli_statistics(self) -> Dict[str, Any]:
        """Get CLI statistics"""
        return self.cli_metrics.copy()

    # Private helper methods

    async def _register_default_commands(self) -> None:
        """Register default CLI commands"""

        # Help command
        self.commands["help"] = self._cmd_help

        # Status command
        self.commands["status"] = self._cmd_status

        # Info command
        self.commands["info"] = self._cmd_info

        # Clear command
        self.commands["clear"] = self._cmd_clear

        # History command
        self.commands["history"] = self._cmd_history

        # Training commands
        self.commands["train"] = self._cmd_train
        self.commands["dataset"] = self._cmd_dataset
        self.commands["preprocess"] = self._cmd_preprocess
        self.commands["model"] = self._cmd_model

    async def _cmd_help(self, args: List[str]) -> str:
        """Handle help command"""
        help_text = """
Available commands:
  help     - Show this help message
  status   - Show system status
  info     - Show system information
  clear    - Clear the screen
  history  - Show command history

For more information about a specific command, type: help <command>
        """
        return help_text.strip()

    async def _cmd_status(self, args: List[str]) -> str:
        """Handle status command"""
        total = max(1, self.cli_metrics["total_commands"])
        successful = self.cli_metrics["successful_commands"]
        success_rate = (successful / total) * 100
        return f"""
System Status:
  CLI Interface: Running
  Commands Executed: {self.cli_metrics["total_commands"]}
  Success Rate: {success_rate:.1f}%
  Session Active: {self.current_session is not None}
        """.strip()

    async def _cmd_info(self, args: List[str]) -> str:
        """Handle info command"""
        return """
SloughGPT CLI Interface
Version: 2.0.0
Description: Advanced AI system command-line interface
        """.strip()

    async def _cmd_clear(self, args: List[str]) -> str:
        """Handle clear command"""
        # Clear screen (platform specific)
        import os

        os.system("cls" if os.name == "nt" else "clear")
        return "Screen cleared"

    async def _cmd_history(self, args: List[str]) -> str:
        """Handle history command"""
        if not self.command_history:
            return "No command history available"

        history_text = "Command History:\n"
        for i, cmd in enumerate(self.command_history[-10:], 1):
            history_text += f"  {i}. {cmd}\n"

        return history_text.strip()

    async def _cmd_train(self, args: List[str]) -> str:
        """Handle train command"""
        from ...training import (
            PipelineConfig,
            TrainingPipeline,
        )

        if not args:
            return "Usage: train <name> [--epochs N] [--batch-size N]\nStarts a training pipeline"

        pipeline_name = args[0]
        epochs = 3
        batch_size = 32

        for i, arg in enumerate(args):
            if arg == "--epochs" and i + 1 < len(args):
                epochs = int(args[i + 1])
            elif arg == "--batch-size" and i + 1 < len(args):
                batch_size = int(args[i + 1])

        config = PipelineConfig(
            name=pipeline_name,
            batch_size=batch_size,
            epochs=epochs,
        )
        _ = TrainingPipeline(config)

        self.logger.info(f"Training pipeline '{pipeline_name}' created")
        return f"Training pipeline '{pipeline_name}' configured:\n  Epochs: {epochs}\n  Batch size: {batch_size}\nUse 'dataset' and 'preprocess' commands to prepare data"

    async def _cmd_dataset(self, args: List[str]) -> str:
        """Handle dataset command"""

        if not args:
            return "Usage: dataset <subcommand> [options]\nSubcommands: list, add, load"

        subcommand = args[0] if args else "list"

        if subcommand == "list":
            return "Available datasets:\n  (no datasets registered)\nUse 'dataset add <name> <path> <type>' to add a dataset"
        elif subcommand == "add" and len(args) >= 4:
            name = args[1]
            path = args[2]
            dtype = args[3]
            return f"Dataset '{name}' would be registered from: {path}\nType: {dtype}"
        else:
            return f"Unknown subcommand: {subcommand}\nUse: list, add"

    async def _cmd_preprocess(self, args: List[str]) -> str:
        """Handle preprocess command"""
        from ...training import DataPreprocessor

        if not args:
            return "Usage: preprocess <dataset> [options]\nOptions: --clean, --filter, --min-length N"

        dataset_name = args[0]
        preprocessor = DataPreprocessor()

        if "--clean" in args:
            preprocessor.add_cleaning()
        if "--filter" in args:
            min_len = 10
            for i, arg in enumerate(args):
                if arg == "--min-length" and i + 1 < len(args):
                    min_len = int(args[i + 1])
            preprocessor.add_filter(min_length=min_len)

        self.logger.info(f"Preprocessor configured for '{dataset_name}'")
        return f"Preprocessing pipeline configured for '{dataset_name}':\n  Steps: {len(preprocessor.steps)}\nUse 'train' to run training"

    async def _cmd_model(self, args: List[str]) -> str:
        """Handle model command"""

        if not args:
            return "Usage: model <subcommand> [options]\nSubcommands: list, create, info"

        subcommand = args[0]

        if subcommand == "list":
            return "Available models:\n  (no models registered)\nUse 'model create <name> <type>' to create a model"
        elif subcommand == "create" and len(args) >= 3:
            name = args[1]
            mtype = args[2]
            return f"Model '{name}' of type '{mtype}' would be created"
        else:
            return f"Unknown subcommand: {subcommand}\nUse: list, create"

    async def _show_help(self) -> None:
        """Show help information"""
        help_output = await self._cmd_help([])
        print(help_output)

    def _format_as_table(self, data: Any) -> str:
        """Format data as table"""
        if isinstance(data, dict):
            table_lines = []
            for key, value in data.items():
                table_lines.append(f"{key:<20} {value}")
            return "\n".join(table_lines)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)
