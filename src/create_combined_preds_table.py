
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=[
      'google-cloud-bigquery==2.34.4', 
      'google-cloud-aiplatform==1.21.0'
  ],
)
def create_combined_preds_table(
  project: str,
  dataset: str,
  bq_location: str,
  model_1_eval_table_uri: str,
  model_2_eval_table_uri: str,
  model_1_path: Input[Artifact],
  model_2_path: Input[Artifact],
  override: str = 'False',
) -> NamedTuple('Outputs', [
    ('combined_preds_table_uri', str),
    ('eval_bq_dataset', str),
]):
    
    from google.cloud import bigquery

    override = bool(override)
    
    bq_client = bigquery.Client(
        project=project, 
        location="US", # bq_location
    )
    
    combined_preds_table_name = f'{project}.{dataset}.combined_preds'

    model_1_eval_table_uri=model_1_eval_table_uri
    model_2_eval_table_uri=model_2_eval_table_uri

    def _sanitize_bq_uri(bq_uri):
        if bq_uri.startswith("bq://"):
            bq_uri = bq_uri[5:]
        return bq_uri.replace(":", ".")

    model_1_eval_table_uri = _sanitize_bq_uri(
        model_1_eval_table_uri
    )

    model_2_eval_table_uri = _sanitize_bq_uri(
        model_2_eval_table_uri
    )

    (
        bq_client.query(
            f"""
            CREATE {'OR REPLACE TABLE' if override else 'TABLE IF NOT EXISTS'}
                `{combined_preds_table_name}`
            AS (
                SELECT * except(row_number) from
                (
                  SELECT *,ROW_NUMBER() OVER (PARTITION BY datetime,vertex__timeseries__id order by predicted_on_date asc) row_number
                  FROM
                (
                  SELECT
                  DATE(table_a.date) as datetime,
                  DATE(table_a.predicted_on_date) as predicted_on_date,
                  CAST(table_a.gross_quantity as INTEGER) as gross_quantity,
                  table_a.vertex__timeseries__id,
                  table_a.predicted_gross_quantity.value as predicted_gross_quantity_a,
                  table_b.predicted_gross_quantity.value as predicted_gross_quantity_b
                  FROM
                  `{model_1_eval_table_uri}` AS table_a
                  INNER JOIN `{model_2_eval_table_uri}` AS table_b
                  ON DATE(table_a.date) = DATE(table_b.date)
                  and table_a.vertex__timeseries__id = table_b.vertex__timeseries__id
                  and DATE(table_a.predicted_on_date) = DATE(table_b.predicted_on_date)
                ) a
              )m
              where row_number = 1
            );
          """
        )
        .result()
    )

    return (
        f'bq://{combined_preds_table_name}',
        f'{dataset}',
    )
