# Forecasting with Vertex Tabular Workflows 

> TODO

### L2L pipeline
<img src='imgs/l2l-full-pipe-run.png'>

### TiDE pipeline
<img src='imgs/tide-e2e-pipeline.png'>


### Code Snippets

#### Retrieve the uploaded Vertex model with a Vertex Pipeline job id

**Example format of pipeline_job_id:** `projects/540160140086/locations/us-central1/pipelineJobs/tide-forecasting-de601790-4a65-400d-96f8-a22c0b6756a1`

```python
pipeline_job_id` = "projects/540160140086/locations/us-central1/pipelineJobs/tide-forecasting-de601790-4a65-400d-96f8-a22c0b6756a1"
job = aiplatform.PipelineJob.get(pipeline_job_id)
pipeline_task_details = job.gca_resource.job_detail.task_details
upload_model_task = get_task_detail(
    pipeline_task_details, "model-upload-2"
)

forecasting_mp_model_artifact = (
    upload_model_task.outputs["model"].artifacts[0]
)
forecasting_mp_model = aiplatform.Model(forecasting_mp_model_artifact.metadata['resourceName'])
print(forecasting_mp_model)
```

#### Upload with parent model for different model versions

```python
parent_model_resource_name = ""
parent_model_artifact = aiplatform.Artifact.get_with_uri("https://us-central1-aiplatform.googleapis.com/v1/" + parent_model_resource_name)
parent_model_artifact_id = parent_model_artifact.gca_resource.name.split("artifacts/")[1]

train_budget_milli_node_hours = 250.0  # 15 minutes

(
    template_path,
    parameter_values,
) = automl_forecasting_utils.get_time_series_dense_encoder_forecasting_pipeline_and_parameters(
    project=PROJECT_ID,
    location=REGION,
    root_dir=root_dir,
    target_column=target_column,
    optimization_objective=optimization_objective,
    transformations=transformations,
    train_budget_milli_node_hours=train_budget_milli_node_hours,
    # Do not set `data_source_csv_filenames` and
    # `data_source_bigquery_table_path` if you want to use Vertex managed
    # dataset by commenting out the following two lines.
    data_source_csv_filenames=data_source_csv_filenames,
    data_source_bigquery_table_path=data_source_bigquery_table_path,
    weight_column=weight_column,
    predefined_split_key=predefined_split_key,
    training_fraction=training_fraction,
    validation_fraction=validation_fraction,
    test_fraction=test_fraction,
    num_selected_trials=5,
    time_column=time_column,
    time_series_identifier_columns=[time_series_identifier_column],
    time_series_attribute_columns=time_series_attribute_columns,
    available_at_forecast_columns=available_at_forecast_columns,
    unavailable_at_forecast_columns=unavailable_at_forecast_columns,
    forecast_horizon=forecast_horizon,
    context_window=context_window,
    dataflow_subnetwork=dataflow_subnetwork,
    dataflow_use_public_ips=dataflow_use_public_ips,
    run_evaluation=False,
    # evaluated_examples_bigquery_path=f'bq://{PROJECT_ID}.eval',
    dataflow_service_account=SERVICE_ACCOUNT,
    # Quantile forecast, TiDE & L2L without probabilistic inference requires
    # as `minimize-quantile-loss` as the optimization objective.
    # quantiles=[0.25, 0.5, 0.9],
)

job_id = "tide-forecasting-with-parent-model-{}".format(uuid.uuid4())
job = aiplatform.PipelineJob(
    display_name=job_id,
    location=REGION,  # launches the pipeline job in the specified region
    template_path=template_path,
    job_id=job_id,
    pipeline_root=root_dir,
    parameter_values=parameter_values,
    enable_caching=False,
    input_artifacts={'parent_model': str(parent_model_artifact_id)},
)
```

#### Integrate Tabular Workflow for Forecasting into your existing KFP pipeline

This is implemented using [the pipeline-as-component feature](https://www.kubeflow.org/docs/components/pipelines/v2/load-and-share-components/) of KFP.

```python
from kfp import dsl
from kfp import compiler
from kfp import components


# Number of weak models in the final ensemble model.
num_selected_trials = 5

train_budget_milli_node_hours = 250.0  # 15 minutes

(
    template_path,
    parameter_values,
) = automl_forecasting_utils.get_time_series_dense_encoder_forecasting_pipeline_and_parameters(
    project=PROJECT_ID,
    location=REGION,
    root_dir=root_dir,
    target_column=target_column,
    optimization_objective=optimization_objective,
    transformations=transformations,
    train_budget_milli_node_hours=train_budget_milli_node_hours,
    data_source_csv_filenames=data_source_csv_filenames,
    data_source_bigquery_table_path=data_source_bigquery_table_path,
    weight_column=weight_column,
    predefined_split_key=predefined_split_key,
    training_fraction=training_fraction,
    validation_fraction=validation_fraction,
    test_fraction=test_fraction,
    num_selected_trials=num_selected_trials,
    time_column=time_column,
    time_series_identifier_columns=[time_series_identifier_column],
    time_series_attribute_columns=time_series_attribute_columns,
    available_at_forecast_columns=available_at_forecast_columns,
    unavailable_at_forecast_columns=unavailable_at_forecast_columns,
    forecast_horizon=forecast_horizon,
    context_window=context_window,
    dataflow_subnetwork=dataflow_subnetwork,
    dataflow_use_public_ips=dataflow_use_public_ips,
    run_evaluation=False,
    dataflow_service_account=SERVICE_ACCOUNT,
)

# Load the forecasting pipeline as a sub-pipeline/components which can be used
# in a larger KFP pipeline.
forecasting_pipeline = components.load_component_from_file(template_path)

@dsl.component
def print_message(msg: str):
  print("message:", msg)


# Define a pipeline that follows the below steps:
# step_1(print_message) -> step_2(print_message) -> forecasting_pipeline
@dsl.pipeline
def outer_pipeline(msg_1: str, msg_2: str, ds: dsl.Artifact):
  step_1 = print_message(msg=msg_1)
  step_2 = print_message(msg=msg_2).after(step_1)
  # `vertex_dataset` argument needs to be set/forwarded here to avoid the
  # "missing-argument" error in KFP pipeline.
  forecasting_pipeline(**parameter_values, vertex_dataset=ds).after(step_2)


# Compile and save the outer/larger pipeline template.
outer_pipeline_template_path = "./outer_pipeline.yaml"
compiler.Compiler().compile(outer_pipeline, outer_pipeline_template_path)


job_id = "run-forecasting-pipeline-inside-pipeline-{}".format(uuid.uuid4())
job = aiplatform.PipelineJob(
    display_name=job_id,
    location=REGION,  # launches the pipeline job in the specified region
    template_path=outer_pipeline_template_path,
    job_id=job_id,
    pipeline_root=root_dir,
    parameter_values={'msg_1': 'step 1', 'msg_2': 'step 2'},
    enable_caching=False,
)

job.run(service_account=SERVICE_ACCOUNT)
```