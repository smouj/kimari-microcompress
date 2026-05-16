"""Example: Kimari CLI plugin registration.

This example demonstrates how a Kimari CLI application can
register KMC commands using the plugin interface.
"""

from __future__ import annotations

from kmc.integrations.kimari_plugin import (
    KIMARI_PLUGIN_COMMAND_MAP,
    register_kimari_commands,
)


class MockKimariCLI:
    """Mock Kimari CLI for demonstration purposes."""

    def __init__(self) -> None:
        self.commands: dict[str, object] = {}

    def add_command(self, name: str, func: object) -> None:
        """Register a command with the CLI."""
        self.commands[name] = func

    def show_registered(self) -> None:
        """Print all registered commands."""
        print("Kimari CLI — Registered KMC Commands:")
        for name, func in self.commands.items():
            print(f"  {name}: {func.__name__}")


def main() -> None:
    """Demonstrate Kimari plugin registration."""
    app = MockKimariCLI()

    # Register all KMC commands
    register_kimari_commands(app)

    # Show what was registered
    app.show_registered()

    # Show the command mapping
    print("\nKMC Plugin Command Map:")
    for kimari_cmd, func in KIMARI_PLUGIN_COMMAND_MAP.items():
        print(f"  {kimari_cmd} → {func.__name__}")

    # Demonstrate usage
    print("\nUsage examples:")
    print("  kimari compress ./model ./model.kmc --tensor-aware")
    print("  kimari decompress ./model.kmc ./restored/")
    print("  kimari verify-compress ./model.kmc")
    print("  kimari verify-compress ./model.kmc --quick")
    print("  kimari bench-compress ./model ./model-bench.kmc")
    print("  kimari inspect-model ./model/")


if __name__ == "__main__":
    main()
