
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
def create_final_pred_table(
    project: str,
    dataset: str,
    bq_location: str,
    combined_preds_table_uri: str,
    override: str = 'False',
) -> NamedTuple('Outputs', [
    ('final_preds_table_uri', str),
]):
    from google.cloud import bigquery

    override = bool(override)
    
    bq_client = bigquery.Client(
        project=project, 
        location="US", # bq_location
    )
    
    final_preds_table_name = f'{project}.{dataset}.final_preds'
    
    (
        bq_client.query(
            f"""
            CREATE {'OR REPLACE TABLE' if override else 'TABLE IF NOT EXISTS'}
                `{final_preds_table_name}`
            AS (
              SELECT
                  datetime,
                  vertex__timeseries__id,
                  gross_quantity as gross_quantity_actual,
                  ROUND(a.predicted_gross_quantity_a, 2) as model_a_pred,
                  ROUND(a.predicted_gross_quantity_b, 2) as model_b_pred,
                  ROUND((a.predicted_gross_quantity_a + a.predicted_gross_quantity_b)/2, 2) AS Final_Pred,
                  ROUND(ABS(gross_quantity - ((a.predicted_gross_quantity_a + a.predicted_gross_quantity_b)/2)), 2) as Final_Pred_error,
              FROM
                `{combined_preds_table_uri[5:]}` AS a);
            """
        )
        .result()
    )

    return (
      f'bq://{final_preds_table_name}',
    )
