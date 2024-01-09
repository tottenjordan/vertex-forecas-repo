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

def args_generate_ints(
    experiment_dict: str,
) -> int:
# ) -> NamedTuple('Outputs', [
#     ('cw_value', int),
# ]):
    import json
    import logging
    
    logging.info(f'experiment_dict: {experiment_dict}')
    
    entry_dump = json.loads(experiment_dict)
    logging.info(f'experiment_dict: {experiment_dict}')
    
    integer_value_cw = int(entry_dump['context_window'])
    
    return integer_value_cw
