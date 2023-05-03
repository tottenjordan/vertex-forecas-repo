# Pipeline workflows

contents
* `simple-pipeline` - getting familiar with orchestrating forecast workflows
* `experiment-pipeline` - how to use pipelines to train and evaluate multiple model configs
* `end-to-end-pipeline` - full workflow including, data prep, training, evaluation, custom metrics, and forecasting future val
---
## simple pipeline

* this pipeline illustrates basic components of a Vertex Forecast workflow
* run this to understand how to package the SDK examples into a managed pipeline

**steps:**
* create BigQuery dataset
> data prep & validation
> * create input table specs
> * validate forecast data
> * preprocessing forecast data
> * prepare forecast data for train job 
* create a Vertex Managed Dataset
* Train Vertex Forecast AutoML job
> * interpret model evaluation metrics

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/imgs/vf-simple-pipe-complete.png)

---
## experiment-pipeline

* TODO

**steps:**
* TODO

<!-- ![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/imgs/vf-simple-pipe-complete.png) -->

---
## end-to-end-pipeline

* TODO

**steps:**
* TODO

<!-- ![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/imgs/vf-simple-pipe-complete.png) -->