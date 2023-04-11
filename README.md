# vertex-forecas-repo
create RDF pipeline workflows with Cloud Source Repositories, Cloud Shell, etc. 

* see the [`knowledge-share`](https://github.com/tottenjordan/vertex-forecas-repo/tree/main/knowledge-share) folder for discussion on select topics

## getting started...

### (1) Create [Vertex Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction) instance; clone repo

* open a terminal and clone this repository to your instance:

`git clone https://github.com/tottenjordan/vertex-forecas-repo.git`

### (2) Install packages

```bash
pip install -U google-cloud-storage --user
pip install kfp==1.8.19 --user
pip install google_cloud_pipeline_components==1.0.41 --user
pip install--U google-cloud-aiplatform==1.22.0 --user
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
