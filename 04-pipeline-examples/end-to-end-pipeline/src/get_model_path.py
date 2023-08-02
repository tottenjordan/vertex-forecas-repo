
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
def get_model_path(
    project: str,
    location: str,
    model: Input[Artifact],
) -> NamedTuple('Outputs', [
    ('model_resource_path', str)
]):
    
    from google.cloud import aiplatform as vertex_ai
    import json
    import logging

    vertex_ai.init(
        project=project,
        location=location,
    )

    model_resource_path = model.metadata["resourceName"]
    logging.info("model path: %s", model_resource_path)
    
    return (
        model_resource_path,
    )
