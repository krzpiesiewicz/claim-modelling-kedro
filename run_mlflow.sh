#!/bin/bash
source venv/bin/activate && cd mlflow && mlflow server --host 127.0.0.1 --port 5000
