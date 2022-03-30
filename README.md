# Production Machine Learning <br> Demo For Dyson
By Lau Wen Hao

In this demo, I will be going through the end to end process of a Machine Learning Lifecycle to demonstrate machine learning systems in production to the best of my knowledge, discussing the processes and tools that may be involved.

I will be assuming that there is already a notebook service and data platform (data lake, datawarehouse, ETL framework, etc.) available and will only focus on discussing the machine learning related systems. I believe as usage of machine learning scales, the operations needed to manage these in production will scale exponentially, as the systems become more and more complex to support the changing business requirements. Hence developing a unified platform for data scientists to self service would be essential for growth.

## Requirements to run
- Docker
- Docker Compose
- Python

## Usage
To run:
```
$ pip install -r requirements.txt
$ docker-compose up --build -d
```

This will deploy the MLflow server and MinIO server that will be used in the notebook later.
Ports used:
- 3000 - MLflow Model Server (created from the notebook)
- 5000 - MLflow server
- 9000 - MinIO API server
- 9001 - MinIO Console

After these are up and running, you can run the notebook cell by cell.

Teardown:
```
$ docker stop mlflow-serving
$ docker-compose down
```

<hr>

## 1. Problem Statement

Every Data Science project starts with defining the problem statement and the success metric. In this project, I will be using the [New York City Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv) dataset to try to predict the price of Airbnb listing in New York given the host id, location (neighourhood group, latitude, longitude), and room type.

## 2. Model Development
In the `airbnb-predict-price.ipynb` notebook, I went through the steps of data exploration, cleaning and producing a simple Linear Regression model. The dataset is cleaned and splitted into the training and testing set. The model performance is measured by the r2 score on the testing set.

The resulting model and metric are logged to the locally deployed [MLflow tracking server](https://www.mlflow.org/docs/latest/tracking.html).

In reality, data scientists will go through numerous iterations to build the best models, experimenting with feature engineering, different model architectures, different parameters, hyperparamter tuning, etc.

<b>Experiment Tracking</b>

Data sciencists are encouraged to track their experiments as they develop their machine learning models as it helps them to keep their experiments organised and reproducible. MLflow has been chosen as the tool to do this as it provides a simple and clean interface to achieve this and also provide other tools in the machine learning lifecycle in the later steps.

## 3. Model Registry

Once data scientists are satisfied with the models, productionalisation would be the next step, i.e. deploy the model to use it in a production use case. The first step to do this is to register this model into a registry. This is akin to developers pushing new code into their GitHub repository when they are done with development.

The model registry provides teams with visibility over their models, with the ability to version control, and provide the lineage on which dataset, what parameters, the model produced was trained on. It should also provide the history/metadata on when this model is produced, when and where it was deployed. I believe model registry to be a central component of MLOps that enables the data scientists, software developers and operational teams to collaborate. 

The model registry chosen is the one provided by [MLflow](https://www.mlflow.org/docs/latest/model-registry.html) as it integrates nicely with the tracking service. The Databricks managed MLflow provides some other integrations within their platform, for example to deploy an endpoint, but the open source MLflow model registry left it open for organizations to implement their own hooks and processes for approvals and integrate into their existing CICD platform.

For example, registering a new model can trigger a pipeline to build a new Docker image and deploy this model as a service, which will be covered in more details in the next section.

## 4. Model Deployment
I consider there to be mainly 3 kinds of usage pattern of a machine learning model.

### 4.1. Embedded into application logic
The simplest way is to embed models into the application logic. Models will be treated as artifacts in the deployment of the application. Most data science projects start off from this approach when the team and the application is still small and more integrated as it is the fastest and easiest to get features/products up and running and there is no network latency involved.

However, as the team and application scales, and more people are involved, this method might not be feasible in the long run as it would incur a lot of technical debt when models are updated frequently or when multiple models and experiments are involved as updating models requires the redeployent of the whole application.

Another use case is when the model is required to run on an edge or mobile device that may not have network connection.

### 4.2. Real-Time Prediction via API (a.k.a Dynamic Serving)
Deploying as a standalone service, usually as a microservice part of a service mesh. This is usually the way to go for services that requires real time predictions based on user inputs, for example a recommendation systems reacting to user's activity on an e-commerce platform.

There are multiple tools available in the market that can achieve this, MLflow, [KServe (formerly KFServing)](https://github.com/kserve/kserve), [Seldon Core](https://github.com/SeldonIO/seldon-core), etc. The one that I chose is again [MLflow](https://www.mlflow.org/docs/latest/models.html) for serving, which provides a simple API for the serving the model and it supports most of the popular machine learning frameworks used in the industry.

However this simple approach might not be the best solution. MLflow runs the different types of model in its native framework with a wrapper which might not be the most optimized if high QPS is a requirement. 

[ONNX (Open Neural Network Exchange)](https://onnx.ai/) is an open format built to represent machine learning models, an open standard for machine learning interoperability. Models trained from different frameworks can be easily converted into this format to be served in any environment. After conversion, there are tools like [ONNX Runtime](https://github.com/microsoft/onnxruntime) that provides optimizations to the computational graphs to run these models.

This coupled with [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) can give inference speed a massive performance boost.

### 4.3. Batch Job Prediction (a.k.a Static Serving)
This is usually for predictions that are not time sensitive, for example a scan job to flag out spam. Jobs can run on a cron schedule, hourly, daily, weekly, etc. The most common tool for scheduling and managing such jobs is [Apache Airflow](https://airflow.apache.org/), which is an open source workflow management platform, primarily for data engineering pipelines.

Another emerging tool in this space is [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/introduction/), which uses [Argo Workflows](https://argoproj.github.io/workflows/) underneath which runs on Kubernetes.

## 5. Monitoring

Once the services are up and running, the next step would be monitor these in production. There are mainly two types of monitoring that is involved.

### 5.1 Service Monitoring
As Kubernetes becomes more widely adopted by organisation, most open source software provide prometheus exporters that exposes the custom metrics that can be scraped by a Prometheus server to centralise monitoring.

In my experience, I find [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) the most comprehensive as it simplifies and automates a lot of the configurations required to set up the full Prometheus stack (including alert managers and visualisation tool like grafana) for monitoring Kubernetes Clusters. It also provides easy APIs and CRDs to extend this monitoring to any applications running on the cluster.

Through these tools, we will be able to monitor the model's traffic patterns, error rates, latency, and resource utilization. This can help spot problems with the models and also find the right machine type to optimize latency and cost.

### 5.2 Model Performance Monitoring

As a best practice, there is also a need for a way to actively monitor the quality of the models in production. Monitoring lets us detect model performance degradation or model staleness. The outputs of monitoring for these changes then feeds into the data analysis component, which could serve as a trigger to execute the retraining pipeline or to execute a new experimental cycle.

Monitoring should be designed to detect data skews which occur when the model training data is not representative of the live data. The data that we use to train the model does not represent the data that we actually receive in the live traffic, and this leads to model staleness. Production data can diverge or drift from the baseline data over time due to changes in the real world. 

The more common types of drift includes:
- Data Drift - a shift in the model's input data distribution. 
- Concept Drift - a shift in the actual relationship between the model inputs and output. 

## 6. Continuous Training

Continuous training is the process where models are trained and updated on fresh data either by a trigger or a scheduled job in production through an <b>automated</b> pipeline.

<b>Scheduled Jobs</b>
Training pipelines can be set to run on a monthly, weekly or even daily basis depending on the business scenarios.

<b>Triggered Jobs</b>
This can be integrated with the model performance monitoring component. These triggers can be things like the aforementioned drifts or when performance of the model drops below ceertain threshold.

These can be achieved with the same tools used by the batch prediction pipelines, Airflow, Kubeflow Pipelines or Argo. However, there needs to be a robust process to handle the change management as the entire training pipeline is deployed to run automated in production.

## 7. Live Experiment

The last component I will touch on will be the setup that allows live experimentation, also known as online validation. Offline validation should be standard and part of the model development stage. This is to validate the trained model to determine if it should be promoted into production.

Production deployment of a model usually goes through A/B testing or live experiments on live traffic before the model is promoted to serve all the prediction request.

It is natively supported and a feature in KServe and Seldon to be able to switch the traffic on the fly between models using the CRDs that comes with deployment of KServe and Seldon.

## 8. Conclusion
I think these are the most important stages in a machine learning lifecycle. There are certainly much more that I can explore, things like Feature Store and Metadata Store but I have not yet understood the use cases and the need for them. To end off, below are some new topics that I've come across in the MLOps space which are interesting.

[MLOps: From Model-centric to Data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo)
> MLOps' most important task: Ensure consistently high-quality data in all phases of the ML project lifecycle.

[Open MRM Project](https://github.com/openMRM/OpenMRM/)
> an open source software project for a composable Model Risk Management (MRM) services architecture with the explicit purpose of sharing MRM best practices for managing Machine Learning (ML) models.

[MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

I only came across this document after I've completed this assignment. Google Cloud has laid out clear definitions on the different levels of MLOps capabilities which I find is a good measure for any organisation trying to build a Machine Learning Platform.
