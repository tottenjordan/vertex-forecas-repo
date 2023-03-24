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

```bash
XXX
```

