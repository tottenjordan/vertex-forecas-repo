# FastFresh demo

### TODOs
* use same dataset to build BQARIMA+, Vertex Forecast, FB Prophet 
* include simple notebook for each
* include example that orchestrates all steps with Vertex Pipelines
* implement ML Evaluation job pipeline

### repo structure
`bqml-pipeline-demand-forecast.ipynb` 
> * Defines a custom evaluation component
> * Defines a pipeline:
>> * Gets BigQuery training data
>> * Trains a BigQuery Arima Plus model
>> * Evaluates the BigQuery Arima Plus model
>> * Plots the evaluations
>> * Checks the model performance
> * Generates the ARIMA Plus forecasts
> * Generates the ARIMA PLUS forecast explainations
> * Compiles & executes pipeline shown below

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/bq-arima-pipeline.png)

#### model evaluation metrics 
* this pipelines `plot_evaluation_metrics` component creates an HTML pipeline `Artifact` describing the evaluation metrics 
* in the pipelines UI, follow the output artifact's link to view the HTML file in your browser (should look like the table below)

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/repo-imgs/bq-arima-pipe-output-metrics.png)