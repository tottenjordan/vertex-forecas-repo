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
* prepares and preprocesses forecast train and eval datasets
* train multiple Vertex Forecast models, each with different training configurations (e.g.,`optimization-objective == mape | rmse`)
* conditional (optional) pipeline workflows:
> * **(1) evaluate models as an ensemble:** combine (average) each trained model's forecasts on `TEST` set
> * **(2) evaluate models individually:** run model evaluation workflow for each model, import eval metric artifacts to model object in Vertex AI Model Registry
> * **(3) forecast models on `FORECAST PLAN`:** for each model, run batch prediction job for future dates (`FORECAST PLAN`); combines predictions in single table for downstream production tasks

---
# Pipeline explained:

> The pipeline notebook creates an end-to-end forecasting pipeline with conditional arguments to control which tasks are performed during each pipeline run

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/overall-pipeline-collapsed-conditonals.png)

## [1] Evaluate trained models as ensemble

pipeline argument    |  tasks | Pipeline |
|:-------------------------------:|:----------------------  |:------------------------- |
`combine_preds_flag == True`     | <ul><li>creates combines prediction table</li><li>averages predictions and calculates ABS forecast error</li></ul>| ![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/overall-pipe-expanded-condition-1.png)   |

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/combine-forecasts-on-test-set.png)

---
## [2] evaluate models individually

pipeline argument    |  tasks | Pipeline |
:-------------------------------:|:-------------------------:|:------------------------- |
`model_evaluation_flag == True`  | prepares Model evaluation dataset | ![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/overall-pipe-collapsed-condition-2.png) |

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/prepare-seperate-model-eval-cond.png)

## [2.X] control which trained models are evaluated 

pipeline argument    |  tasks | Pipeline |
:-------------------------------:|:-------------------------:|:------------------------- |
`{MODEL_TYPE}_eval_flag == True`  | runs model evaluation pipeline for specified models | ![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/overall-pipe-expanded-condition-2.png)

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/seperate-model-eval-cond.png)

### creates [Dataflow](https://cloud.google.com/dataflow) job to run [Tensorflow Model Analysis](https://www.tensorflow.org/tfx/model_analysis/get_started) (TFMA) for forecast evaluation

* `ModelEvaluationForecastingOp()` creates Dataflow job to run Tensorflow Model Analysis (TFMA) for forecast evaluation
* TFMA job produces a `google.ForecastingMetrics` Artifact; 
* artifact uploaded to model metadata in Vertex Model Registry enables eval comparison in Vertex console UI

Dataflow job running TFMA  |  Compare models in the Vertex console UI
:---------------:|:--------:|
![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/dataflow-tfma-eval-job.png)  | ![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/model-eval-vertex-ui.png)

---
## [3] conditional: `forecast_plan_flag == True`
### forecast models on `FORECAST PLAN`

pipeline argument    |  tasks | Pipeline |
:-------------------------------:|:-----------------------  |:-------------------------:
`combine_preds_flag == True`     | batch prediction job for each model to forecast future (unseen) demand | ![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/overall-pipe-explanded-condition-3.png)

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/forecast-plan-conditional.png)

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
