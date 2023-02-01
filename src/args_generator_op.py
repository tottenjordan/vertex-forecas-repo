
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
def args_generator_op(
    project: str,
    location: str,
    rmse_model: Input[Artifact],
    rmse_display_name: str,
    mape_model: Input[Artifact],
    mape_display_name: str,
) -> str:

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

    return json.dumps(
        [
            {
                "model": rmse_model_resource_path, 
                "objective": "rmse", 
                "display_name": rmse_display_name
            }, 
            {
                "model": mape_model_resource_path, 
                "objective": "mape", 
                "display_name": mape_display_name
            }
        ],
        sort_keys=True,
    )



#     # create nested JSON list (string)
#     model_versions = json.dumps(
#         [
#             {
#                 "model": f"{get_rmse_model_path.outputs['model_resource_path']}",
#                 "display_name": rmse_model_version,
#                 "optimization_objective": "rmse",
#             },
#             {
#                 "model": f"{get_mape_model_path.outputs['model_resource_path']}",
#                 "display_name": mape_model_version,
#                 "optimization_objective": "mape",
#             },
#         ],
#         sort_keys=True,
#     )
    
    
#         mape_model_version = f'{VERSION}-seq2seq-mape' # TODO: determines model display name and eval BQ table name # f'{VERSION}-l2l-mape'
#     rmse_model_version = f'{VERSION}-seq2seq-rmse'
