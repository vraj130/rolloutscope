"""Command-line interface for rolloutscope.

Thin wrapper only: parses arguments and delegates to the library packages. The full
command set (analyze, convert, detectors list, schema export) arrives in Phase 6; this
module exists from Phase 1 so the console-script entry point resolves.
"""

import typer

from rolloutscope import __version__

app = typer.Typer(
    no_args_is_help=True,
    help="Offline rollout and reward-hacking debugger for the verifiers / prime-rl ecosystem.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the rolloutscope version and exit.",
    ),
) -> None:
    """rolloutscope: analyze rollout artifacts for reward hacking, offline."""
