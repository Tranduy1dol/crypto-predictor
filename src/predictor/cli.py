import typer

from .predictor import predictor

app = typer.Typer()
app.command()(predictor)

if __name__ == "__main__":
    app()
