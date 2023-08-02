
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image='python:3.9'
)
def create_input_table_specs(
    products_table_uri: str,
    activities_table_uri: str,
    locations_table_uri: str,
    time_granularity_unit: str,
    time_granularity_quantity: int,
    # train_data_bq_source: str,
) -> NamedTuple('Outputs', [
    ('input_table_specs', str),
    ('model_feature_columns', str),
    # ('train_data_source_bq_uri', str),
]):
    
    import json
    import logging
    logging.getLogger().setLevel(logging.INFO)

    # train_data_source_bq_uri = f'bq://{train_data_bq_source}'

    products_table_specs = {
        'bigquery_uri': products_table_uri,
        'table_type': 'FORECASTING_ATTRIBUTE',
        'forecasting_attribute_table_metadata': {
            'primary_key_column': 'product_id'
        }
    }

    locations_table_specs = {
        'bigquery_uri': locations_table_uri,
        'table_type': 'FORECASTING_ATTRIBUTE',
        'forecasting_attribute_table_metadata': {
            'primary_key_column': 'location_id'
        }
    }

    activities_table_specs = {
        'bigquery_uri': activities_table_uri,
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
    }

    model_feature_columns = [
        'product_id',
        'location_id',
        'gross_quantity',
        'date',
        'weekday',
        'wday',
        'month',
        'year',
        'event_name_1',
        'event_type_1',
        'event_name_2',
        'event_type_2',
        'snap_CA',
        'snap_TX'
        'snap_WI',
        'dept_id',
        'cat_id',
        'state_id',
    ]

    input_table_specs = [
    activities_table_specs,
    products_table_specs,
    locations_table_specs,
    ]

    return (
        json.dumps(input_table_specs),  # input_table_specs
        json.dumps(model_feature_columns),  # model_feature_columns
        # f'{train_data_source_bq_uri}',
    )
