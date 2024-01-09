import kfp
from typing import NamedTuple
from kfp.dsl import (
    # Artifact, 
    # Dataset, 
    # Input, InputPath, 
    # Model, Output, OutputPath, 
    component, 
    Metrics
)
@component(
  base_image='python:3.10',
  packages_to_install=['google-cloud-bigquery==3.14.1'],
)
def create_bq_dataset(
    project: str,
    new_bq_dataset: str,
    bq_location: str
) -> NamedTuple('Outputs', [
    ('bq_dataset_name', str),
    ('bq_dataset_uri', str),
]):
    
    from google.cloud import bigquery

    bq_client = bigquery.Client(project=project, location=bq_location) # bq_location)
    (
      bq_client.query(f'CREATE SCHEMA IF NOT EXISTS `{project}.{new_bq_dataset}`')
      .result()
    )
    
    return (
        f'{new_bq_dataset}',
        f'bq://{project}.{new_bq_dataset}',
    )
