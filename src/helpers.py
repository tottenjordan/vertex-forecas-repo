import os
import sys
import uuid
import json
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform, storage

# Fetch the tuple of GCS bucket and object URI.
def get_bucket_name_and_path(uri):
    no_prefix_uri = uri[len("gs://") :]
    splits = no_prefix_uri.split("/")
    return splits[0], "/".join(splits[1:])

# Fetch the content from a GCS object URI.

def download_from_gcs(uri):
    bucket_name, path = get_bucket_name_and_path(uri)
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.download_as_string()

# Upload the string content as a GCS object.

def write_to_gcs(uri: str, content: str):
    bucket_name, path = get_bucket_name_and_path(uri)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_string(content)


def generate_auto_transformation(column_names: List[str]) -> List[Dict[str, Any]]:
    transformations = []
    for column_name in column_names:
        transformations.append({"auto": {"column_name": column_name}})
    return transformations

# This is the example to set non-auto transformations.
# For more details about the transformations, please check:
# https://cloud.google.com/vertex-ai/docs/datasets/data-types-tabular#transformations
def generate_transformation(
    auto_column_names: Optional[List[str]] = None,
    numeric_column_names: Optional[List[str]] = None,
    categorical_column_names: Optional[List[str]] = None,
    text_column_names: Optional[List[str]] = None,
    timestamp_column_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if auto_column_names is None:
        auto_column_names = []
    if numeric_column_names is None:
        numeric_column_names = []
    if categorical_column_names is None:
        categorical_column_names = []
    if text_column_names is None:
        text_column_names = []
    if timestamp_column_names is None:
        timestamp_column_names = []
    return {
        "auto": auto_column_names,
        "numeric": numeric_column_names,
        "categorical": categorical_column_names,
        "text": text_column_names,
        "timestamp": timestamp_column_names,
    }


def write_auto_transformations(uri: str, column_names: List[str]):
    transformations = generate_auto_transformation(column_names)
    write_to_gcs(uri, json.dumps(transformations))

# Retrieve the data given a task name.
def get_task_detail(
    task_details: List[Dict[str, Any]], task_name: str
) -> List[Dict[str, Any]]:
    for task_detail in task_details:
        if task_detail.task_name == task_name:
            return task_detail

# Retrieve the URI of the model.
def get_deployed_model_uri(
    task_details,
):
    ensemble_task = get_task_detail(task_details, "model-upload")
    return ensemble_task.outputs["model"].artifacts[0].uri


def get_no_custom_ops_model_uri(task_details):
    ensemble_task = get_task_detail(task_details, "automl-tabular-ensemble")
    return download_from_gcs(
        ensemble_task.outputs["model_without_custom_ops"].artifacts[0].uri
    )

# Retrieve the feature importance details from GCS.
def get_feature_attributions(
    task_details,
):
    ensemble_task = get_task_detail(task_details, "model-evaluation-2")
    return download_from_gcs(
        ensemble_task.outputs["evaluation_metrics"]
        .artifacts[0]
        .metadata["explanation_gcs_path"]
    )

# Retrieve the evaluation metrics from GCS.
def get_evaluation_metrics(
    task_details,
):
    ensemble_task = get_task_detail(task_details, "model-evaluation")
    return download_from_gcs(
        ensemble_task.outputs["evaluation_metrics"].artifacts[0].uri
    )

# Pretty print the JSON string.
def load_and_print_json(s):
    parsed = json.loads(s)
    print(json.dumps(parsed, indent=2, sort_keys=True))