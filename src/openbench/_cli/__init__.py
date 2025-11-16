import typer
from openbench._cli.eval_command import run_eval
from openbench._cli.eval_retry_command import run_eval_retry
from openbench._cli.list_command import list_evals
from openbench._cli.describe_command import describe_eval
from openbench._cli.view_command import run_view
from openbench._cli.cache_command import cache_app
from openbench._cli.export_command import run_export

app = typer.Typer(rich_markup_mode="rich")

app.command("list")(list_evals)
app.command("describe")(describe_eval)
app.command("eval")(run_eval)
app.command("eval-retry")(run_eval_retry)
app.command("view")(run_view)
app.command("export-hf")(run_export)
app.add_typer(cache_app, name="cache")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
