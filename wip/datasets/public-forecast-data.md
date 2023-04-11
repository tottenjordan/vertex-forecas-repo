# Publicly available datasets

## FastFresh synthetic dataset
* used in tutorial: [Build and visualize demand forecast predictions using Datastream, Dataflow, BigQuery ML, and Looker](https://cloud.google.com/architecture/build-visualize-demand-forecast-prediction-datastream-dataflow-bigqueryml-looker)
* tutorial intended for data engineers and analysts who want to use their operational data
* fictitious retail store named FastFresh to help demonstrate selling fresh produce
* **business objectives**
> * minimize food waste 
> * optimize stock levels across all stores

## Google Analytics 4 ecommerce web implementation
* BigQuery Public Datasets program
* Google Merchandise Store selling Google-branded merchandise. 
* site uses Google Analytics 4's standard web ecommerce implementation along with enhanced measurement 
* The `ga4_obfuscated_sample_ecommerce` dataset contains a sample of obfuscated BigQuery event export data for three months from `2020-11-01` to `2021-01-31`

### EDA example code

**get the data**

```sql
TODO
```

**unique events, users, and days** in the dataset

```sql
SELECT
  COUNT(*) AS event_count,
  COUNT(DISTINCT user_pseudo_id) AS user_count,
  COUNT(DISTINCT event_date) AS day_count
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
```


### Retail demand dataset
* data covers 10 US stores 
* feature set includes
> * item level, 
> * department, 
> * product categories, 
> * store details
* explanatory variables e.g., price and gross margin
* [guided steps](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/231b2ef02b5f902b167b37ac3d36c64800c054d8/notebooks/official/workbench/demand_forecasting/forecasting-retail-demand.ipynb) for data wrangling

```sql
SELECT * FROM looker-private-demo.retail.transaction_detail
```

### NYC Bike Trips
* BigQuery Public Dataset program
* EDA and data prep [guide](https://github.com/statmike/vertex-ai-mlops/blob/main/Applied%20Forecasting/1%20-%20BigQuery%20Time%20Series%20Forecasting%20Data%20Review%20and%20Preparation.ipynb) from a trusted friend, @statmike !!