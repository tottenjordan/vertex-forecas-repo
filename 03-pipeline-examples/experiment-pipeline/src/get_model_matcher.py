
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
def get_model_matcher(
    project: str,
    location: str,
    model_display_name: str,
    version: str,
    experiment_name: str,
    data_regime: str,
# ) -> str:
) -> NamedTuple('Outputs', [
    ('parent_model_resource_name', str)
]):

    from google.cloud import aiplatform as vertex_ai
    import logging

    vertex_ai.init(
        project=project,
        location=location,
    )
    
    modelmatch = vertex_ai.Model.list(filter = f'display_name={model_display_name} AND labels.data_regime={data_regime} AND labels.experiment={experiment_name}')

    if modelmatch:
        logging.info("There is an existing model with versions: ", [f'{m.version_id}' for m in modelmatch])
        parent = modelmatch[0].resource_name
    else:
        logging.info("This is the first training for this model")
        parent = 'None'
        
    return (
        f'{parent}',
    )
