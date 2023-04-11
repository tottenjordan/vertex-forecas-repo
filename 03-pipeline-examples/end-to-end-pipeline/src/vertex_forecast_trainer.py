
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image='python:3.9',
    packages_to_install=[
        'google-cloud-aiplatform==1.22.1'
    ],
)
def vertex_forecast_trainer(
    project: str,
    location: str,
    version: str,
    experiment_name: str,
    eval_destination_dataset: str,
    data_regime: str,
    model_backbone: str,
    model_display_name: str,
    optimization_objective: str,
    column_specs:  list, #List[str], #list,
    target_column: str, #List[str], #str,
    time_column: str, #List[str], #str,
    time_series_identifier_column: str, #List[str], #str,
    time_series_attribute_columns: list, #List[str], #str,
    unavailable_at_forecast_columns: list, #List[str], #str,
    available_at_forecast_columns: list, #List[str], #str,
    predefined_split_column_name: str,
    forecast_horizon: int,
    data_granularity_unit: str,
    data_granularity_count: int,
    context_window: int,
    holiday_regions: list,
    hierarchy_group_columns: list,
    hierarchy_group_total_weight: float,
    hierarchy_temporal_total_weight: float,
    hierarchy_group_temporal_total_weight: float,
    # export_evaluated_data_items_bigquery_destination_uri: str,
    budget_milli_node_hours: int,
    parent_model_resource_name: str,
    # column_specs_dict_gcs_uri: str,
    # managed_dataset_resource_name: str,
    managed_dataset_resource_name: Input[Artifact],
) -> NamedTuple('Outputs', [
    # ('model_display_name', str),
    ('model_resource_name', str),
    # ('bq_eval_uri', str)
]):
    
    from google.cloud import aiplatform as vertex_ai
    import json
    import logging
    import pickle as pkl
    from datetime import datetime
    
    EXPERIMENT_NAME = experiment_name.replace("-","_")

    vertex_ai.init(
        project=project,
        location=location,
        experiment=experiment_name.replace("_","-")
    )
    
    BQ_EVAL_DESTINATION_URI = f"bq://{project}:{eval_destination_dataset}:{EXPERIMENT_NAME}_{model_backbone}_eval"
    logging.info(f"BQ_EVAL_DESTINATION_URI: {BQ_EVAL_DESTINATION_URI}")
    MANAGED_DATASET_RESOURCE_NAME = managed_dataset_resource_name.metadata["resourceName"]
    logging.info(f"MANAGED_DATASET_RESOURCE_NAME: {MANAGED_DATASET_RESOURCE_NAME}")
    
    # json array convert type
    # column_specs:  List[str], #list,
    # target_column: List[str], #str,
    # time_column: List[str], #str,
    # time_series_identifier_column: List[str], #str,
    # time_series_attribute_columns: List[str], #str,
    # unavailable_at_forecast_columns: List[str], #str,
    # available_at_forecast_columns: List[str], #str,
    
    logging.info(f"column_specs: {column_specs}")
    logging.info(f"target_column: {target_column}")
    logging.info(f"time_column: {time_column}")
    logging.info(f"time_series_identifier_column: {time_series_identifier_column}")
    logging.info(f"time_series_attribute_columns: {time_series_attribute_columns}")
    logging.info(f"unavailable_at_forecast_columns: {unavailable_at_forecast_columns}")
    logging.info(f"available_at_forecast_columns: {available_at_forecast_columns}")
    
    # column_specs = json.loads(str(column_specs))
    # logging.info(f"column_specs: {column_specs}")
    # ======================================
    # managed dataset reference
    # ======================================
    dataset = vertex_ai.TimeSeriesDataset(MANAGED_DATASET_RESOURCE_NAME)
    
    # ======================================
    # define model architecture backbone
    # ======================================
    if model_backbone == 'seq2seq':
        forecasting_job = vertex_ai.SequenceToSequencePlusForecastingTrainingJob(
            display_name = f'train-{model_display_name}',
            optimization_objective = optimization_objective,
            # column_specs = column_specs,
            column_transformations = column_specs, 
            labels = {'data_regime' : f'{data_regime}', 'experiment' : f'{experiment_name}'}
        )
    elif model_backbone == 'tft':
        forecasting_job = vertex_ai.TemporalFusionTransformerForecastingTrainingJob(
            display_name = f'train-{model_display_name}',
            optimization_objective = optimization_objective,
            # column_specs = column_specs,
            column_transformations = column_specs,
            labels = {'data_regime' : f'{data_regime}', 'experiment' : f'{experiment_name}'}
        )
    elif model_backbone == 'l2l':
        forecasting_job = vertex_ai.AutoMLForecastingTrainingJob(
            display_name = f'train-{model_display_name}',
            optimization_objective = optimization_objective,
            # column_specs = column_specs,
            column_transformations = column_specs,
            labels = {'data_regime' : f'{data_regime}', 'experiment' : f'{experiment_name}'}
        )
    else:
        logging.info("Must provide 1 of 3 model_types: seq2seq, tft, l2l,")
        
    # ======================================
    # execute model train job
    # ======================================
    forecast = forecasting_job.run(
        # data parameters
        dataset = dataset,
        target_column = target_column,
        time_column = time_column,
        time_series_identifier_column = time_series_identifier_column,
        time_series_attribute_columns = time_series_attribute_columns,
        unavailable_at_forecast_columns = unavailable_at_forecast_columns,
        available_at_forecast_columns = available_at_forecast_columns,
        predefined_split_column_name = predefined_split_column_name,

        # forecast parameters
        forecast_horizon = forecast_horizon,
        data_granularity_unit = data_granularity_unit,
        data_granularity_count = data_granularity_count,
        context_window = context_window,
        holiday_regions = holiday_regions,

        hierarchy_group_columns = hierarchy_group_columns,
        hierarchy_group_total_weight = hierarchy_group_total_weight,
        hierarchy_temporal_total_weight = hierarchy_temporal_total_weight,
        hierarchy_group_temporal_total_weight = hierarchy_group_temporal_total_weight,

        # output parameters
        export_evaluated_data_items = True,
        export_evaluated_data_items_bigquery_destination_uri=BQ_EVAL_DESTINATION_URI,
        export_evaluated_data_items_override_destination = True,

        # running parameters
        validation_options = "fail-pipeline",
        budget_milli_node_hours = budget_milli_node_hours,

        # model parameters
        model_display_name = f"{model_display_name}",
        model_labels = {'data_regime' : f'{data_regime}', 'experiment' : f'{experiment_name}'},
        # model_id = f"model_{SERIES}_{EXPERIMENT}",
        # parent_model = parent_model_resource_name,
        is_default_version = True,

        # session parameters: False means continue in local session, True waits and logs progress
        sync = True
    )
    
    # trained model
    MODEL_DISPLAY_NAME = forecast.display_name
    RESOURCE_NAME = forecast.resource_name
    logging.info(f"MODEL_DISPLAY_NAME: {MODEL_DISPLAY_NAME}")
    logging.info(f"RESOURCE_NAME: {RESOURCE_NAME}")
    
    # ======================================
    # default model eval metrics
    # ======================================
    model_evaluation = list(forecast.list_model_evaluations())[0]
    metrics_dict = {k: [v] for k, v in dict(model_evaluation.metrics).items()}
    logging.info(f"metrics_dict: {metrics_dict}")
    
    # ======================================
    # log metrics to Vertex Experiments
    # ======================================
    
    # create run name
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_RUN_NAME = f"run-{TIMESTAMP}"
    
    # log params and metrics to dicts
    params = {}
    params["m_type"] = model_backbone
    params["opt_objective"] = optimization_objective
    params["horizon"] = forecast_horizon
    params["context_window"] = context_window
    metrics = {}
    metrics["MAE"] = metrics_dict['meanAbsoluteError'][0]
    metrics["RMSE"] = metrics_dict['rootMeanSquaredError'][0]
    metrics["MAPE"] = metrics_dict['meanAbsolutePercentageError'][0]
    metrics["rSquared"] = metrics_dict['rSquared'][0]
    metrics["RMSLE"] = metrics_dict['rootMeanSquaredLogError'][0]
    metrics["WAPE"] = metrics_dict['weightedAbsolutePercentageError'][0]

    with vertex_ai.start_run(EXPERIMENT_RUN_NAME) as my_run:
        my_run.log_metrics(metrics)
        my_run.log_params(params)

        vertex_ai.end_run()
        
    return (
        # MODEL_DISPLAY_NAME,
        RESOURCE_NAME,
        # BQ_EVAL_DESTINATION_URI,
    )
