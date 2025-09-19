import typer
from openbench._cli.eval_command import run_eval
from openbench._cli.eval_retry_command import run_eval_retry
from openbench._cli.list_command import list_evals
from openbench._cli.describe_command import describe_eval
from openbench._cli.view_command import run_view
from openbench._cli.copilot_prepare_command import run_copilot_prepare

app = typer.Typer(rich_markup_mode="rich")

app.command("list")(list_evals)
app.command("describe")(describe_eval)
app.command("eval")(run_eval)
app.command("eval-retry")(run_eval_retry)
app.command("view")(run_view)
app.command("mcp-copilot-prepare")(run_copilot_prepare)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
