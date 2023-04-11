
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(base_image='python:3.9')
def get_eval_dataset_path_uri(
    project: str,
    eval_bq_dataset: str,
    model_1_table: str,
    model_2_table: str,
) -> NamedTuple('Outputs',[
    ('model_1_bigquery_table_uri', str),
    ('model_2_bigquery_table_uri', str),
    ('eval_bq_dataset', str),
]):
    
    import json
    import logging

    model_1_table_path_name = f'{project}:{eval_bq_dataset}:eval-{model_1_table}'
    model_2_table_path_name = f'{project}:{eval_bq_dataset}:eval-{model_2_table}'

    logging.info(model_1_table_path_name)
    logging.info(model_2_table_path_name)

    return (
        f'bq://{model_1_table_path_name}',
        f'bq://{model_2_table_path_name}',
        f'{eval_bq_dataset}',
    )
