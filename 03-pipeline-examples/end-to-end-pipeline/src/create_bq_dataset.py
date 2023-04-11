
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.34.4'],
)
def create_bq_dataset(
    project: str,
    vertex_dataset: str,
    new_bq_dataset: str,
    bq_location: str
) -> NamedTuple('Outputs', [
    ('bq_dataset_name', str),
    ('bq_dataset_uri', str),
]):
    
    from google.cloud import bigquery

    bq_client = bigquery.Client(project=project, location='US') # bq_location)
    (
      bq_client.query(f'CREATE SCHEMA IF NOT EXISTS `{project}.{new_bq_dataset}`')
      .result()
    )
    
    return (
        f'{new_bq_dataset}',
        f'bq://{project}:{new_bq_dataset}',
    )
