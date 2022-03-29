FROM python:3.8-slim-bullseye

RUN pip install mlflow awscli boto3 sklearn

ADD entrypoint.sh .
ADD serve_entrypoint.sh .

RUN chmod +x entrypoint.sh
RUN chmod +x serve_entrypoint.sh

# CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "s3://mlflow", "--host", "0.0.0.0"]
ENTRYPOINT [ "./entrypoint.sh" ]
