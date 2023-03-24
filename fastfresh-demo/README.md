# fastfresh dtutorial

> This tutorial uses a fictitious retail store named FastFresh to help demonstrate the concepts that it describes. FastFresh specializes in selling fresh produce, and wants to minimize food waste and optimize stock levels across all stores. You use mock sales transactions from FastFresh as the operational data in this tutorial

* see full tutorial [here](https://cloud.google.com/architecture/build-visualize-demand-forecast-prediction-datastream-dataflow-bigqueryml-looker#objectives)

1. open terminal window, define environment variables:

```bash
export PROJECT_NAME="hybrid-vertex"
export PROJECT_ID="hybrid-vertex"
export PROJECT_NUMBER="934903580331"
export BUCKET_NAME="${PROJECT_ID}-oracle_retail"
export GCP_ZONE='us-central1-a'
export GCP_NETWORK_NAME='custom-vpc-1'
export GCP_SUBNET_NAME=custom-vpc-1-subnet-uscentral1
```

2. set project

```bash
gcloud config set project ${PROJECT_ID}
```

3. clone github repo

```
git clone https://github.com/caugusto/datastream-bqml-looker-tutorial.git datastream-orcale
```

4. extract transactions to load in Oracle

```bash
cd datastream-bqml-looker-tutorial

bunzip2 sample_data/oracle_data.csv.bz2
```

5. Create a sample Oracle XE 11g docker instance on Compute Engine:
* Creates a new Google Cloud Compute instance.
* Configures an Oracle 11g XE docker container.
* Pre-loads the FastFresh schema and the Datastream prerequisites.

> note: may need to update naming conventions in `build_orcl.sh`

```bash
cd build_docker

./build_orcl.sh \
 -p $PROJECT_NAME \
 -z $GCP_ZONE \
 -n $GCP_NETWORK_NAME \
 -s $GCP_SUBNET_NAME \
 -f Y \
 -d Y
```

**After the script executes**
* the `build_orcl.sh` script gives you a summary of the connection details and credentials (DB Host, DB Port, and SID)
* Make a copy of these details; will use them later

6. Create a Cloud Storage bucket to store your replicated data:

```bash
gsutil mb gs://${BUCKET_NAME}
```

7. Configure bucket to send notifications about object changes to a Pub/Sub topic (required by the Dataflow template)

* Create a new topic called `oracle_retail`
> * new topic, `oracle_retail`, sends notifications about object changes to the Pub/Sub topic

```bash
gsutil notification create -t projects/${PROJECT_ID}/topics/oracle_retail -f json gs://${BUCKET_NAME}
```

* Create Pub/Sub subscription to receive messages sent to `oracle_retail` topic

```bash
gcloud pubsub subscriptions create oracle_retail_sub --topic=projects/${PROJECT_ID}/topics/oracle_retail
```

8. Create a BigQuery dataset named `retail`:

```bash
bq mk --dataset ${PROJECT_ID}:retail
```

9. Assign the BigQuery Admin role to your Compute Engine service account:

```bash
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member=serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com --role='roles/bigquery.admin'
```

## Replicate Oracle data to Google Cloud with Datastream

### TODO

## Create a Dataflow job using the Datastream to BigQuery template

1. copy and save the following code to a file named `retail_transform.js`

```js
function process(inJson) {

   var obj = JSON.parse(inJson),
   includePubsubMessage = obj.data && obj.attributes,
   data = includePubsubMessage ? obj.data : obj;

   data.PAYMENT_METHOD = data.PAYMENT_METHOD.split(':')[0].concat("XXX");

   data.ORACLE_SOURCE = data._metadata_schema.concat('.', data._metadata_table);

   return JSON.stringify(obj);
}
```
2. upload the JavaScript file to the newly created bucket

```bash
gsutil mb gs://js-${BUCKET_NAME}

gsutil cp retail_transform.js \
gs://js-${BUCKET_NAME}/utils/retail_transform.js
```

### Create a Dataflow job

```bash
gsutil mb gs://dlq-${BUCKET_NAME}
```

2. Create a service account for the Dataflow execution

```bash
gcloud iam service-accounts create df-tutorial

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
--member="serviceAccount:df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/dataflow.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
--member="serviceAccount:df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/dataflow.worker"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
--member="serviceAccount:df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/pubsub.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
--member="serviceAccount:df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
--member="serviceAccount:df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
--member="serviceAccount:df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/datastream.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
--member="serviceAccount:df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--role="roles/storage.admin"
```
3. Create a firewall egress rule to let Dataflow VMs communicate, send, and receive network traffic on TCP ports 12345 and 12346 when autoscale is enabled

```bash
gcloud compute firewall-rules create fw-allow-inter-dataflow-comm \
--action=allow \
--direction=ingress \
--network=$GCP_NETWORK_NAME  \
--target-tags=dataflow \
--source-tags=dataflow \
--priority=0 \
--rules tcp:12345-12346
```
4. Create and run a Dataflow job:

```bash
export REGION=us-central1
gcloud dataflow flex-template run orders-cdc-template --region ${REGION} \
--template-file-gcs-location "gs://dataflow-templates/latest/flex/Cloud_Datastream_to_BigQuery" \
--service-account-email "df-tutorial@${PROJECT_ID}.iam.gserviceaccount.com" \
--parameters \
inputFilePattern="gs://${BUCKET_NAME}/",\
gcsPubSubSubscription="projects/${PROJECT_ID}/subscriptions/oracle_retail_sub",\
inputFileFormat="json",\
outputStagingDatasetTemplate="retail",\
outputDatasetTemplate="retail",\
deadLetterQueueDirectory="gs://dlq-${BUCKET_NAME}",\
autoscalingAlgorithm="THROUGHPUT_BASED",\
mergeFrequencyMinutes=1,\
javascriptTextTransformGcsPath="gs://js-${BUCKET_NAME}/utils/retail_transform.js",\
javascriptTextTransformFunctionName="process"
```

> Check the Dataflow console to verify that a new streaming job has started.

5. In Cloud Shell, run the following command to start your Datastream stream:

```bash
gcloud datastream streams update oracle-cdc \
--location=us-central1 --state=RUNNING --update-mask=state
```
6. Check the Datastream status

```bash
gcloud datastream streams list \
--location=us-central1
```

> Validate that the state shows as `Running`. It may take a few seconds for the new state value to be reflected.

> Check the Datastream console to validate the progress of the ORDERS table backfill.
