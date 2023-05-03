
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
def model_batch_prediction_job(
    project: str,
    location: str,
    eval_bq_dataset: str,
    bigquery_source: str,
    model_name: str,
    model_1_path: Input[Artifact],
) -> NamedTuple('Outputs', [
    ('batch_predict_output_bq_uri', str),
    ('batch_predict_job_dict', dict)
]):

    from google.cloud import aiplatform as vertex_ai
    import json
    import logging

    vertex_ai.init(
        project=project,
        location=location,
    )

    model_resource_path = model_1_path.metadata["resourceName"]
    logging.info("model path: %s", model_resource_path)

    model = vertex_ai.Model(model_name=model_resource_path)
    logging.info("Model dict:", model.to_dict())

    batch_predict_job = model.batch_predict(
        bigquery_source=bigquery_source,
        instances_format="bigquery",
        bigquery_destination_prefix=f'bq://{project}.{eval_bq_dataset}',
        predictions_format="bigquery",
        job_display_name=f'bpj-{model_name}',
    )

    batch_predict_bq_output_uri = "{}.{}".format(
      batch_predict_job.output_info.bigquery_output_dataset,
      batch_predict_job.output_info.bigquery_output_table
    )

    logging.info(batch_predict_job.to_dict())
    
    return (
        batch_predict_bq_output_uri,
        batch_predict_job.to_dict(),
    )
