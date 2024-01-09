"""
Dynamically generates a Data Studio link that specifies the template, the location of the forecasts, and the query to generate the chart. 
   The data is populated from the forecasts generated using BigQuery options where the destination dataset is batch_predict_bq_output_dataset_path
   You can inspect the used template at https://datastudio.google.com/c/u/0/reporting/067f70d2-8cd6-4a4c-a099-292acd1053e8. 

Note: The Data Studio dashboard can only show the charts properly when the batch_predict job is run successfully using the BigQuery options.

Used in example here: https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/automl/sdk_automl_tabular_forecasting_batch.ipynb
"""

import urllib

tables = client.list_tables(batch_predict_bq_output_dataset_path)

prediction_table_id = ""
for table in tables:
    if (
        table.table_id.startswith("predictions_")
        and table.table_id > prediction_table_id
    ):
        prediction_table_id = table.table_id
batch_predict_bq_output_uri = "{}.{}".format(
    batch_predict_bq_output_dataset_path, prediction_table_id
)


def _sanitize_bq_uri(bq_uri):
    if bq_uri.startswith("bq://"):
        bq_uri = bq_uri[5:]
    return bq_uri.replace(":", ".")


def get_data_studio_link(
    batch_prediction_bq_input_uri,
    batch_prediction_bq_output_uri,
    time_column,
    time_series_identifier_column,
    target_column,
):
    batch_prediction_bq_input_uri = _sanitize_bq_uri(batch_prediction_bq_input_uri)
    batch_prediction_bq_output_uri = _sanitize_bq_uri(batch_prediction_bq_output_uri)
    base_url = "https://datastudio.google.com/c/u/0/reporting"
    query = (
        "SELECT \n"
        " CAST(input.{} as DATETIME) timestamp_col,\n"
        " CAST(input.{} as STRING) time_series_identifier_col,\n"
        " CAST(input.{} as NUMERIC) historical_values,\n"
        " CAST(predicted_{}.value as NUMERIC) predicted_values,\n"
        " * \n"
        "FROM `{}` input\n"
        "LEFT JOIN `{}` output\n"
        "ON\n"
        "CAST(input.{} as DATETIME) = CAST(output.{} as DATETIME)\n"
        "AND CAST(input.{} as STRING) = CAST(output.{} as STRING)"
    )
    query = query.format(
        time_column,
        time_series_identifier_column,
        target_column,
        target_column,
        batch_prediction_bq_input_uri,
        batch_prediction_bq_output_uri,
        time_column,
        time_column,
        time_series_identifier_column,
        time_series_identifier_column,
    )
    params = {
        "templateId": "067f70d2-8cd6-4a4c-a099-292acd1053e8",
        "ds0.connector": "BIG_QUERY",
        "ds0.projectId": PROJECT_ID,
        "ds0.billingProjectId": PROJECT_ID,
        "ds0.type": "CUSTOM_QUERY",
        "ds0.sql": query,
    }
    params_str_parts = []
    for k, v in params.items():
        params_str_parts.append('"{}":"{}"'.format(k, v))
    params_str = "".join(["{", ",".join(params_str_parts), "}"])
    return "{}?{}".format(base_url, urllib.parse.urlencode({"params": params_str}))


print(
    get_data_studio_link(
        PREDICTION_DATASET_BQ_PATH,
        batch_predict_bq_output_uri,
        time_column,
        time_series_identifier_column,
        target_column,
    )
)