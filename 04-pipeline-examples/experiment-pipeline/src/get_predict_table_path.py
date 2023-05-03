
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image='python:3.9'
)
def get_predict_table_path(
    predict_processed_table: str
) -> NamedTuple('Outputs', [
    ('preprocess_bq_uri', str)
]):
    
    import json
    import logging

    logging.info(f"predict_processed_table: {predict_processed_table}")
    
    preprocess_bq_uri = (
        json.loads(predict_processed_table)
        ['processed_bigquery_table_uri']
    )
    
    logging.info(f"preprocess_bq_uri: {preprocess_bq_uri}")
    
    return (
        f'{preprocess_bq_uri}',
    )
