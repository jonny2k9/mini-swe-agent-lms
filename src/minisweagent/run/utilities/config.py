#!/usr/bin/env python3

"""Utility to manage the global config file.

You can also directly edit the `.env` file in the config directory.

It is located at [bold green]{global_config_file}[/bold green].
"""

import os
import subprocess

from dotenv import load_dotenv, set_key, unset_key
from rich.console import Console
from rich.rule import Rule
from typer import Argument, Typer

from minisweagent import global_config_file


def _reload_config():
    load_dotenv(dotenv_path=global_config_file, override=True)


app = Typer(
    help=__doc__.format(global_config_file=global_config_file),  # type: ignore
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console(highlight=False)


_SETUP_HELP = """To get started, we need to set up your [bold green]LM Studio[/bold green] connection.

You can edit settings manually or use the [bold green]mini-extra config set[/bold green] or [bold green]mini-extra config edit[/bold green] commands.

This setup will ask for your LM Studio API endpoint, an optional API key, and the default model.

[bold]LM Studio default endpoint:[/bold] [bold green]http://localhost:1234[/bold green]
[bold]API key:[/bold] LM Studio accepts any non-empty string (leave blank to use the default).
[bold]Model:[/bold] Enter the model name as shown in LM Studio (e.g., [bold green]lmstudio-community/qwen2.5-7b-instruct[/bold green]).

[bold yellow]You can leave any setting blank to keep the current/default value.[/bold yellow]

More information at https://mini-swe-agent.com/latest/quickstart/
"""


def prompt(*args, **kwargs):
    # Defer import to avoid slow import module
    from prompt_toolkit.shortcuts.prompt import prompt as _prompt

    return _prompt(*args, **kwargs)


def configure_if_first_time():
    if not os.getenv("MSWEA_CONFIGURED"):
        console.print(Rule())
        setup()
        console.print(Rule())


@app.command()
def setup():
    """Setup the global config file for LM Studio."""
    console.print(_SETUP_HELP.format(global_config_file=global_config_file))
    endpoint = prompt(
        "Enter your LM Studio API endpoint: ",
        default=os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234"),
    ).strip()
    if endpoint:
        set_key(global_config_file, "LMSTUDIO_ENDPOINT", endpoint)
    api_key = prompt(
        "Enter your LM Studio API key (leave blank for default 'lm-studio'): ",
        default=os.getenv("LMSTUDIO_API_KEY", ""),
    ).strip()
    if api_key:
        set_key(global_config_file, "LMSTUDIO_API_KEY", api_key)

    # Try to list models from LM Studio to help the user pick one
    try:
        from minisweagent.models.lmstudio_model import LMStudioModel

        models = LMStudioModel.list_models(endpoint or os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234"))
        if models:
            console.print("\n[bold green]Available models in LM Studio:[/bold green]")
            for i, m in enumerate(models, 1):
                console.print(f"  {i}. {m}")
            console.print()
    except Exception:
        console.print("[bold yellow]Could not connect to LM Studio to list models. Is it running?[/bold yellow]")

    default_model = prompt(
        "Enter your default model name: ",
        default=os.getenv("LMSTUDIO_MODEL", os.getenv("MSWEA_MODEL_NAME", "")),
    ).strip()
    if default_model:
        set_key(global_config_file, "LMSTUDIO_MODEL", default_model)
        set_key(global_config_file, "MSWEA_MODEL_NAME", default_model)

    set_key(global_config_file, "MSWEA_CONFIGURED", "true")
    _reload_config()
    console.print(
        "\n[bold yellow]Config finished.[/bold yellow] If you want to revisit it, run [bold green]mini-extra config setup[/bold green]."
    )


@app.command()
def set(
    key: str | None = Argument(None, help="The key to set"),
    value: str | None = Argument(None, help="The value to set"),
):
    """Set a key in the global config file."""
    if key is None:
        key = prompt("Enter the key to set: ")
    if value is None:
        value = prompt(f"Enter the value for {key}: ")
    set_key(global_config_file, key, value)
    _reload_config()


@app.command()
def unset(key: str | None = Argument(None, help="The key to unset")):
    """Unset a key in the global config file."""
    if key is None:
        key = prompt("Enter the key to unset: ")
    unset_key(global_config_file, key)
    _reload_config()


@app.command()
def edit():
    """Edit the global config file."""
    editor = os.getenv("EDITOR", "nano")
    subprocess.run([editor, global_config_file])
    _reload_config()


if __name__ == "__main__":
    app()
