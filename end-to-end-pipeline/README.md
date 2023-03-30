# e2e Pipeline explained:

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