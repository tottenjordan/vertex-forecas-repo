# vertex-forecas-repo
create RDF pipeline workflows with Cloud Source Repositories, Cloud Shell, etc. 

### Install these packages
```
pip install -U google-cloud-storage --user
pip install kfp==1.8.18 --user
pip install google_cloud_pipeline_components==1.0.36 --user
pip install--U google-cloud-aiplatform==1.21.0 --user
```


### saving for later
```
# from google_cloud_pipeline_components.types import artifact_types

print("model_1_path:", model_1_path)
model_aip_uri=model_1_path["uri"]

print("model_1_path[49:]:", model_1_path[49:])
model_aip_uri=model_1_path[49:]

print("model_aip_uri_2:", model_aip_uri)
model = artifact_types.VertexModel(uri='xxx')
```

```
if batch_predict_bq_output_uri.startswith("bq://"):
    batch_predict_bq_output_uri = batch_predict_bq_output_uri[5:]

batch_predict_bq_output_uri.replace(":", ".")
```
