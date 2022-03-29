#!/bin/bash
set -e

mlflow models serve -m /data/1/${MLFLOW_RUN_ID}/artifacts/model -h 0.0.0.0 -p 3000 --no-conda