# vertex-forecas-repo
create RDF pipeline workflows with Cloud Source Repositories, Cloud Shell, etc. 

### (1) follow data download instructions in `m5_dataprep` notebook

* run cells to prepare the dataset for demand forecasting

### (2) Create [Vertex Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction) instance; clone repo

* open a terminal and clone this repository to your instance:

`git clone https://github.com/tottenjordan/vertex-forecas-repo.git`

### (3) Install packages

```
pip install -U google-cloud-storage --user
pip install kfp==1.8.18 --user
pip install google_cloud_pipeline_components==1.0.36 --user
pip install--U google-cloud-aiplatform==1.21.0 --user
pip install gcsfs==2023.1.0 --user
```

### (4) Run `vf-demand-pipeline-m5` notebook
* notebook creates pipeline components under `src`
* trains two models, each with different optimization objectives
* evaluates each model, and combines (simple average) their predictions
* batch forecast job on a `PLAN TABLE`


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
