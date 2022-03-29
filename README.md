# Production Machine Learning <br> Demo For Dyson

In this demo, I will be going through the end to end process of a  Machine Learning Lifecycle from to demonstrate my knowledge of machine learning systems in production, discussing the processes and tools that may be involved.

I will be assuming that there is already a notebook service and data platform (data lake, datawarehouse, ETL framework, etc.) available and will only focus on discussing the machine learning related systems in a platform approach. I believe as usage of machine learning scales, the operations needed to manage these in the production will scale exponentially, as systems becomes more and more complex to support the changing business requirements. Hence developing a unified platform for data scientists to self service would be key.

## Usage
```
$ docker-compose up --build -d
```

This will deploy the MLflow server and MinIO server that will be used in the notebook later.

## 1. Problem Statement

Every Data Science project starts with defining the problem statement and the success metric. In this project, I will be using the [New York City Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv) dataset to try to predict the price of Airbnb listing in New York given the ...

## 2. Model Development
In the `airbnb-predict-price.ipynb` notebook, I went through the steps of data exploration, cleaning and producing a simple Linear Regression model. The dataset is cleaned and splitted into the training and testing set. The model performance is measured by the r2 score on the testing set.

The resulting model and metric are logged to the locally deployed MLflow tracking server.

In reality, data scientists will go through numerous iterations to build the best models, experimenting with feature engineering, different model architectures, different parameters, hyperparamter tuning, etc.

<b>Experiment Tracking</b>

Data sciencists are encouraged to track their experiments as they develop their machine learning models as it helps them to keep their experiments organised and reproducible. MLflow has been chosen as the tool to do this as it provides a simple and clean interface to achieve this and also provide other tools in the machine learning lifecycle in the later steps.

## 3. Model Registry

Once data scientists are satisfied with the models, productionalisation would be the next step, i.e. deploy the model to use it in a production use case. The first step to do this is to register this model into a registry. This is akin to developers pushing new code into their GitHub repository when they are done with development.

The model registry provides teams with visibility over their models, with the ability to version control, and provide the lineage on which dataset, what parameters, the model produced was trained on. It should also provide the history on when this model is produced, when and where is it deployed. I believe a model registry to be a central component of ML Ops that enables the data scientists, software developers and operational teams to collaborate. 

The model registry chosen is the one provided by MLflow as it integrates nicely with the tracking service. The Databricks managed MLflow provides some other integrations within their platform, for example to deploy an endpoint, but the open source MLflow model registry left it open for organizations to implement their own hooks and processes for approval and integrate into their existing CICD platform.

For example, registering a new model can trigger a pipeline to build a new Docker image and deploy this model as a service, which will be covered in more details in the next section.

## 4. Model Deployment
I consider there to be mainly 3 kinds of usage pattern of a machine learning model.

### 4.1. Embedded into application
The simplest way is to embed models into the application logic. Models will be treated as artifacts in the deployment of the application. Most data science projects start off from this approach when the team and the application is still small and more integrated as it is the fastest and easiest to get features/products up and running and there is no network latency involved.

However, as the team and application scales, and more people are involved, this method might not be feasible in the long run as it would incur a lot of technical debt when models are updated frequently or when multiple models and experiments are involved as updating models requires the redeployent of the whole application.

### 4.2. Real-Time Prediction via API
Deploying as a standalone service, usually as a microservice part of a service mesh. This is usually the way to go for services that requires real time prediction based on user inputs, like recommendation systems reacting to user's activity on an e-commerce platform.

There are multiple tools available in the market that can achieve this. The one that I chose is again MLflow serving, which provides a simple API for the serving the model and it supports most of the popular machine learning frameworks used in the industry.

However this simple approach might not be the best solution in the market currently. MLflow runs the different types of model in its native framework which might not be the most optimized if high QPS is a requirement. 

[ONNX (Open Neural Network Exchange)](https://onnx.ai/) is an open format built to represent machine learning models, an open standard for machine learning interoperability. Models trained from different frameworks can be easily converted into this format to be served in any environment. After conversion, there are tools like [ONNX Runtime](https://github.com/microsoft/onnxruntime) that provides optimizations to the computational graphs to run these models.

This coupled with NVIDIA Triton Inference Server can give inference speed a massive performance boost.

As graphics processors are increasingly being used to accelerate inference, espeicially with deep learning models, 

### 4.3. Batch Job Prediction
This is usually for predictions that are not time sensitive, for example a scan job to flag out spam. Jobs can run on a cron schedule like . The most common tool for such scheduling and managing such jobs is [Apache Airflow](https://airflow.apache.org/), which is an open source workflow management platform, primarily for data engineering pipelines.

Another emerging tool in this space is [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/introduction/), which uses [Argo Workflows](https://argoproj.github.io/workflows/) underneath which runs on Kubernetes.
### 

## 5. Monitoring
### 5.1 Service Monitoring
Prometheus
### 5.2 Model Performance Monitoring

#### Data Drift
#### Concept Drift

## 6. Continuous Training
### Scheduled Jobs
### Triggered Jobs

## 7. Live Experimentation
### AB Test / Traffic Splitting

## 8. Conclusion
I think I have covered most of the stages in a machine learning lifecycle. To end off, below are some new topics that I've come across in the MLOps space which are kind of interesting.

[MLOps: From Model-centric to Data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo)
> MLOps' most important task: Ensure consistently high-quality data in all phases of the ML project lifecycle.

[Open MRM Project](https://github.com/openMRM/OpenMRM/)
> an open source software project for a composable Model Risk Management (MRM) services architecture with the explicit purpose of sharing MRM best practices for managing Machine Learning (ML) models.