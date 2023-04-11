
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
def args_generator_op_1(
    project: str,
    location: str,
    rmse_model: Input[Artifact],
    rmse_display_name: str,
    mape_model: Input[Artifact],
    mape_display_name: str,
# ) -> str:
) -> NamedTuple('Outputs', [
    ('model_list', list)
]):

    from google.cloud import aiplatform as vertex_ai
    import json
    import logging

    vertex_ai.init(
        project=project,
        location=location,
    )

    rmse_model_resource_path = rmse_model.metadata["resourceName"]
    logging.info(f"rmse_model_resource_path: {rmse_model_resource_path}")
    
    mape_model_resource_path = mape_model.metadata["resourceName"]
    logging.info(f"mape_model_resource_path: {mape_model_resource_path}")
    
    model_list = [rmse_model_resource_path, mape_model_resource_path]
    logging.info(f"model_list: {model_list}")
    
    test_list = [
        {
            "model": rmse_model_resource_path, 
            "objective": "rmse", 
        },
        {
            "model": mape_model_resource_path, 
            "objective": "mape", 
        }
    ]
    logging.info(f"test_list: {test_list}")

    return (
        model_list,
    )
