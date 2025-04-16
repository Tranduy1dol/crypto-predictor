import os

import typer
from typing_extensions import Annotated

from .process import predict, train_and_test


def print_help():
    print("""
Make Prediction CLI
====================

Available commands:
- help     : Show this help message.
- predict  : Run prediction using trained model.
- new      : Initialize a new project setup.
    """)


def predictor(
        command: Annotated[str, typer.Argument(help="Command to run: 'predict' or 'new'.")],
        config_path: Annotated[str, typer.Argument(help="Path to the config file for the model.")] = None,
):
    match command:
        case 'predict':
            if not config_path:
                print("❌ Config path is required for prediction.")
                return
            if not os.path.isfile(config_path):
                print(f"❌ Config file not found at: {config_path}")
                return

            print("📈 Running prediction with the trained model...")
            try:
                predict(config_path)
            except Exception as e:
                print(f"❌ Failed to predictor tomorrow close price: {e}")
            print("✅ Prediction completed (stub).")
        case 'new':
            if not os.path.isfile(config_path):
                print(f"❌ Config file not found at: {config_path}")
                return

            print(f"📁 Creating new model using config: {config_path}")
            try:
                train_and_test(config_path)
            except Exception as e:
                print(f"❌ Failed to create model: {e}")
        case 'help':
            print_help()
        case _:
            print(f"❌ Unknown command: {command}")
