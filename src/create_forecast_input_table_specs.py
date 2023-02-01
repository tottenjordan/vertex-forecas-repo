
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image='python:3.9',
    packages_to_install=[
        'google-cloud-aiplatform==1.21.0'
    ],
)
def create_forecast_input_table_specs(
  project: str,
  forecast_products_table_uri: str,
  forecast_activities_table_uri: str,
  forecast_locations_table_uri: str,
  forecast_plan_table_uri: str,
  time_granularity_unit: str,
  time_granularity_quantity: int,
) -> NamedTuple('Outputs', [
    ('forecast_input_table_specs', str)
]):
    import json
    import os
    import logging
    logging.getLogger().setLevel(logging.INFO)

    forecast_input_table_specs = [
        {
            'bigquery_uri': forecast_plan_table_uri,
            'table_type': 'FORECASTING_PLAN',
        },
        {
            'bigquery_uri': forecast_activities_table_uri,
            'table_type': 'FORECASTING_PRIMARY',
            'forecasting_primary_table_metadata': {
                'time_column': 'date',
                'target_column': 'gross_quantity',
                'time_series_identifier_columns': ['product_id', 'location_id'],
                'unavailable_at_forecast_columns': [],
                'time_granularity': {
                    'unit': time_granularity_unit,
                    'quantity': time_granularity_quantity,
                },
                # 'predefined_splits_column': 'ml_use',
                # 'predefined_split_column': 'ml_use', # model_override
            }
        },
        {
            'bigquery_uri': forecast_products_table_uri,
            'table_type': 'FORECASTING_ATTRIBUTE',
            'forecasting_attribute_table_metadata': {
                'primary_key_column': 'product_id'
            }
        },
        {
            'bigquery_uri': forecast_locations_table_uri,
            'table_type': 'FORECASTING_ATTRIBUTE',
            'forecasting_attribute_table_metadata': {
                'primary_key_column': 'location_id'
            }
        },
    ]

    logging.info(f"forecast_input_table_specs: {forecast_input_table_specs}")

    return (
        json.dumps(forecast_input_table_specs),
    )
