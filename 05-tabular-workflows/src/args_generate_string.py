import kfp
from typing import NamedTuple, List, Dict, Any, Union
from kfp.dsl import (
    # Artifact, 
    # Dataset, 
    # Input, InputPath, 
    # Model, Output, OutputPath, 
    component, 
    Metrics
)
@component(
  base_image='python:3.10',
)

def args_generate_string(
    cw_values: List[int],
    opt_objective: str,
    experiment_name: str,
) -> str:
# ) -> NamedTuple('Outputs', [
#     # ('experiment_list', List[Dict[str, str]]),
#     ('experiment_list', str),
# ]):
    import logging
    import json
    
    logging.info(f'NUM_EXPERIMENTS: {len(cw_values)}')
    
    output_list = []
    
    for cw in cw_values:
        entry = {
            "context_window" : str(cw),
            "objective" : opt_objective,
            "model_display_name" : f"{experiment_name}-{str(cw)}",
        }
        output_list.append(entry)
        
    logging.info(f'output_list: {output_list}')
    
    return json.dumps(
        output_list
    )
