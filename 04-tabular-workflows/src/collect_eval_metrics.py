import kfp
from typing import NamedTuple
from kfp.dsl import (
    component, 
    Metrics
)
@component(
  base_image='python:3.10',
  packages_to_install=['google-cloud-aiplatform==1.38.1'],
)
def collect_eval_metrics(
    project_id: str,
    experiment_name: str,
    experiment_run: str,
    pipeline_details: list,
    
):
    from google.cloud import aiplatform
    
    # helper functions
    # Retrieve the data given a task name.
    def get_task_detail(
        task_details: List[Dict[str, Any]], 
        task_name: str
    ) -> List[Dict[str, Any]]:
        for task_detail in task_details:
            if task_detail.task_name == task_name:
                return task_detail
            
    # get uploaded model
    upload_model_task = get_task_detail(
        pipeline_task_details, "model-upload-2"
    )
    forecasting_mp_model_artifact = (
        upload_model_task.outputs["model"].artifacts[0]
    )
    forecasting_mp_model = aiplatform.Model(forecasting_mp_model_artifact.metadata['resourceName'])
    print(f"forecasting_mp_model: {forecasting_mp_model}")
    
    # get evaluations
    model_evaluations = forecasting_mp_model.list_model_evaluations()
    
    # Print the evaluation metrics
    for evaluation in model_evaluations:
        evaluation = evaluation.to_dict()
        print("Model's evaluation metrics from training:\n")
        metrics = evaluation["metrics"]
        for metric in metrics.keys():
            print(f"metric: {metric}, value: {metrics[metric]}\n")
    
    print("message:", msg)
