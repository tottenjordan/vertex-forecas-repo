# Forecast with Vertex and BigQueryML on Google Cloud

> this repo provides code examples for various forecasting use-cases using [Vertex Forecast](https://cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/forecasting-train) (deep learning AutoML), BigQuery ML's [ARIMA+](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-time-series) model, and [Vertex Tabular Workflows](https://cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/forecasting) (pipeline orchestration)

## *Note: Currently refactoring repo (12/29/2023)*

* `02-vf-sdk-example/`    - needs updated SDK vertsion
* `03-bqml-sdk-examples/` - should be up-to-date
* `04-pipeline-examples/` - needs updated versions for SDK and pipeline components
* `05-tabular-workflows/` - **current focus:** consolidating examples; using BQ public dataset

## Getting started...

> see the [`knowledge-share`](https://github.com/tottenjordan/vertex-forecas-repo/tree/main/knowledge-share) folder for discussion on select topics

### (1) Create [Vertex Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction) instance; clone repo

* open a terminal and clone this repository to your instance:

`git clone https://github.com/tottenjordan/vertex-forecas-repo.git`

### (2) Install packages

```bash
pip install -U google-cloud-storage --user
pip install kfp==1.8.19 --user
pip install google_cloud_pipeline_components==1.0.41 --user
pip install--U google-cloud-aiplatform==1.23.0 --user
pip install gcsfs==2023.1.0 --user
```

### (3) follow data download instructions in `01-download-m5-data.ipynb` notebook

* download  public dataset and land in BigQuery

### (4) follow data validation and prep in `02-m5-dataprep.ipynb` notebook

* prepare dataset for demand forecasting

### (5) Get familiar with Vertex Forecast capabilities

> these notebooks build on each other, but do not need to be executed in order

* `03-vertex-forecast-train-sdk.ipynb` - simple train and eval example using Vertex AI SDK
* `04-vertex-forecast-experiments.ipynb` - see how to create parallel train jobs for experimentation
* `05-vf-quantiles.ipynb` - understand probabilistic inference and quantile forecasts
* `<place_holder>` - will demonstrate hierarchical aggregation (group and time)

### (6) Learn how to orchestrate all these steps with Vertex Pipelines

* `07-simple-pipeline-vf.ipynb` - simple pipeline for data prep, training, and evaluation
* `end-to-end-pipeline/XX-vf-demand-pipeline-m5.ipynb` - handling complex workflows, ensemble multiple trained models, custom metrics and evaluation, forecast plan table

### Vertex Forecast options

**Model types**
* Time series Dense Encoder (TiDE)
* Temporal Fusion Transformer (TFT)
* AutoML (L2L)
* Seq2Seq+

**Optimization Objectives** ([docs](https://cloud.google.com/vertex-ai/docs/tabular-data/forecasting-parameters#optimization-objectives))

| Objective  | API                      | Use case |
| :--------: | :------------:           | :------------------------------------- |
| RMSE       | `minimize-rmse`          | Minimize root-mean-squared error (RMSE). Captures more extreme values accurately and is less biased when aggregating predictions.Default value. |
| MAE        | `minimize-mae`           | Minimize mean-absolute error (MAE). Views extreme values as outliers with less impact on model. |
| RMSLE      | `minimize-rmsle`         | Minimize root-mean-squared log error (RMSLE). Penalizes error on relative size rather than absolute value. Useful when both predicted and actual values can be large. |
| RMSPE      | `minimize-rmspe`         | Minimize root-mean-squared percentage error (RMSPE). Captures a large range of values accurately. Similar to RMSE, but relative to target magnitude. Useful when the range of values is large. |
| WAPE       | `minimize-wape-mae`      | Minimize the combination of weighted absolute percentage error (WAPE) and mean-absolute-error (MAE). Useful when the actual values are low. |
| QUANTILE   | `minimize-quantile-loss` | Minimize the scaled pinball loss of the defined quantiles to quantify uncertainty in estimates. Quantile predictions quantify the uncertainty of predictions. They measure the likelihood of a prediction being within a range. |


**TiDE on Vertex Tabluar Workflows**
* [src](https://github.com/kubeflow/pipelines/blob/master/components/google-cloud/google_cloud_pipeline_components/preview/automl/forecasting/utils.py#L413)

---
## simple pipeline

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/imgs/vf-simple-pipe-complete.png)

---
#### TODO: saving for later
```
# from google_cloud_pipeline_components.types import artifact_types

print("model_1_path:", model_1_path)
model_aip_uri=model_1_path["uri"]

print("model_1_path[49:]:", model_1_path[49:])
model_aip_uri=model_1_path[49:]

print("model_aip_uri_2:", model_aip_uri)
model = artifact_types.VertexModel(uri='xxx')
```

```
if batch_predict_bq_output_uri.startswith("bq://"):
    batch_predict_bq_output_uri = batch_predict_bq_output_uri[5:]

batch_predict_bq_output_uri.replace(":", ".")
```
