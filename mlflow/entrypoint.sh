#!/bin/bash
set -e

aws --endpoint-url $MLFLOW_S3_ENDPOINT_URL s3 mb s3://mlflow || true

mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination s3://mlflow --serve-artifacts --host 0.0.0.0