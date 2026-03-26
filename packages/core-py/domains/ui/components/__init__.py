"""
UI Components Implementation

This module provides reusable UI components including
forms, buttons, layouts, and interactive elements.
"""

import logging
from typing import Any, Dict, Optional

from ...__init__ import BaseComponent, ComponentException


class UIComponents(BaseComponent):
    """Reusable UI components system"""

    def __init__(self) -> None:
        super().__init__("ui_components")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Component registry
        self.components: Dict[str, Dict[str, Any]] = {}
        self.component_templates: Dict[str, str] = {}

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize UI components"""
        try:
            self.logger.info("Initializing UI Components...")

            # Register default components
            await self._register_default_components()

            self.is_initialized = True
            self.logger.info("UI Components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize UI Components: {e}")
            raise ComponentException(f"UI Components initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown UI components"""
        try:
            self.logger.info("Shutting down UI Components...")
            self.is_initialized = False
            self.logger.info("UI Components shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown UI Components: {e}")
            raise ComponentException(f"UI Components shutdown failed: {e}")

    async def render_component(
        self, component_name: str, props: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render a UI component"""
        try:
            if component_name not in self.components:
                return f"Component '{component_name}' not found"

            component_def = self.components[component_name]
            component_type = component_def["type"]

            # Ensure context is not None
            safe_context = context or {}

            if component_type == "button":
                return await self._render_button(props, safe_context)
            elif component_type == "input":
                return await self._render_input(props, safe_context)
            elif component_type == "form":
                return await self._render_form(props, safe_context)
            elif component_type == "table":
                return await self._render_table(props, safe_context)
            else:
                return f"Unknown component type: {component_type}"

        except Exception as e:
            self.logger.error(f"Failed to render component {component_name}: {e}")
            return f"Error rendering component: {e}"

    async def register_component(self, name: str, component_def: Dict[str, Any]) -> None:
        """Register a custom component"""
        self.components[name] = component_def
        self.logger.info(f"Registered UI component: {name}")

    # Private helper methods

    async def _register_default_components(self) -> None:
        """Register default UI components"""

        # Button component
        self.components["button"] = {
            "type": "button",
            "props": ["text", "onclick", "style", "disabled"],
        }

        # Input component
        self.components["input"] = {
            "type": "input",
            "props": ["type", "placeholder", "value", "required", "style"],
        }

        # Form component
        self.components["form"] = {
            "type": "form",
            "props": ["fields", "onsubmit", "method", "style"],
        }

        # Table component
        self.components["table"] = {
            "type": "table",
            "props": ["data", "columns", "style", "pagination"],
        }

    async def _render_button(self, props: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Render button component"""
        text = props.get("text", "Button")
        onclick = props.get("onclick", "")
        style = props.get("style", "")
        disabled = props.get("disabled", False)

        disabled_attr = "disabled" if disabled else ""

        return f'<button onclick="{onclick}" style="{style}" {disabled_attr}>{text}</button>'

    async def _render_input(self, props: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Render input component"""
        input_type = props.get("type", "text")
        placeholder = props.get("placeholder", "")
        value = props.get("value", "")
        required = props.get("required", False)
        style = props.get("style", "")

        required_attr = "required" if required else ""

        return (
            f'<input type="{input_type}" placeholder="{placeholder}" '
            f'value="{value}" style="{style}" {required_attr}>'
        )

    async def _render_form(self, props: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Render form component"""
        fields = props.get("fields", [])
        onsubmit = props.get("onsubmit", "")
        method = props.get("method", "POST")
        style = props.get("style", "")

        field_html = ""
        for field in fields:
            field_type = field.get("type", "input")
            if field_type == "input":
                field_html += await self._render_input(field, context)
            elif field_type == "textarea":
                field_html += (
                    f'<textarea name="{field.get("name", "")}">{field.get("value", "")}</textarea>'
                )

        return f'<form method="{method}" onsubmit="{onsubmit}" style="{style}">{field_html}</form>'

    async def _render_table(self, props: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Render table component"""
        data = props.get("data", [])
        columns = props.get("columns", [])
        style = props.get("style", "")

        # Generate header
        header_html = "<thead><tr>"
        for column in columns:
            header_html += f"<th>{column.get('title', column.get('key', ''))}</th>"
        header_html += "</tr></thead>"

        # Generate body
        body_html = "<tbody>"
        for row in data:
            body_html += "<tr>"
            for column in columns:
                key = column.get("key", "")
                value = row.get(key, "")
                body_html += f"<td>{value}</td>"
            body_html += "</tr>"
        body_html += "</tbody>"

        return f'<table style="{style}">{header_html}{body_html}</table>'
