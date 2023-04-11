
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image='python:3.9',
    packages_to_install=[
        'google-cloud-bigquery==2.34.4'
  ],
)
def create_combined_preds_forecast_table(
  project: str,
  dataset: str,
  model_1_pred_table_uri: str,
  model_2_pred_table_uri: str,
  override: str = 'False',
) -> NamedTuple('Outputs', [
    ('combined_preds_forecast_table_uri', str)
]):
    
    from google.cloud import bigquery

    override = bool(override)
    bq_client = bigquery.Client(project=project)
    combined_preds_forecast_table_name = f'{project}.{dataset}.combined_preds_forecast'
    (
        bq_client.query(
            f"""
            CREATE {'OR REPLACE TABLE' if override else 'TABLE IF NOT EXISTS'}
                `{combined_preds_forecast_table_name}`
            AS (
              SELECT
                table_a.date as date,
                table_a.vertex__timeseries__id,
                ROUND(table_a.predicted_gross_quantity.value,2) as predicted_gross_quantity_a,
                ROUND(table_b.predicted_gross_quantity.value, 2) as predicted_gross_quantity_b,
                ROUND((table_a.predicted_gross_quantity.value + table_b.predicted_gross_quantity.value)/2, 2) AS Final_Pred
              FROM
                `{model_1_pred_table_uri[5:]}` AS table_a
              INNER JOIN `{model_2_pred_table_uri[5:]}` AS table_b
              ON table_a.date = table_b.date
              AND table_a.vertex__timeseries__id = table_b.vertex__timeseries__id
            );
          """
        )
        .result()
    )

    return (
        f'bq://{combined_preds_forecast_table_name}',
    )
