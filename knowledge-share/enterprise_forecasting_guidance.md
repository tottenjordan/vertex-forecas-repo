# Tips and best practices for planning and executing Enteerprise Forecast workloads

## (1) Forecasting projects are not about using a tool but rather about using your domain knowledge and a tool to make business decisions

> TODO

## (2) Evaluate forecasting with multiple metrics

Usually one metric alone is not enough to evaluate forecast accuracy
* Multiple metrics are required
* Metrics must be scaled metrics (plain RMSE will not do. You must scale it or normalize it to compare against multiple datasets)
* Metrics must be analyzed by sub segments (a single global metric for the entire company is not a valid metric!)

## (3) Allocate more time than usual for EDA and insights
* Since forecasting is all about business, you must plan to spend more time than usual (40% instead of 20% of project time) in EDA and deriving insights about the customers’ data. 
* Your insights about their data will likely lead to better features, so your time spent will lead to better feature engineering in preparation for models.

## (4) Perform EDA and filling missing gaps
* Spend time understanding your data (e.g. tables and columns); then create hypotheses and validate them with the data. Every plot you make should inform how you model later.
* Sometimes time series dates have gaps. Decide whether you need to “zero fill” these gaps or fill them using “backward fill” and/or “forward fill” strategies.

## (5) Categorize your demand patterns

If categorized data does not already align with existing features/columns, create features which categorize them appropriately.

<img src='imgs/cateogrize_demand_patterns.png'>

* Sparse vs Non-sparse: Consider separating sparse and non-sparse data if categories are very distinct
* Sudden vs smooth: Create features for smooth & predictable trends, investigate sudden events for outliers, correcting if anomaly
* Intermittent vs non-intermittent: Understand which category your data falls into, and create appropriate optimization/evaluation metrics
* Seasonal vs non-seasonal: Create explicit features which fit to seasonality (can be naive like flags, or complex like sin functions); investigate non-seasonality for cause and correct if necessary
* In some cases, forecast accuracy is greatly improved by removing extremely sparse time series that are essentially unforecastable from the dataset

See this paper, for more on this topic: [categorization of demand patterns](https://www.jstor.org/stable/4102103)

## (6) Establish a core set of features first
* All product hierarchy attributes (e.g., item, category, and division descriptions, etc.)
* A few broad location attributes (e.g., urban vs rural, pop. density, floor plan rather than ZIP, name, etc.)
* Sales (target), inventory (quantity or OOS flag)

## (7) Use unavailable-at-forecast covariates to teach the model about the past

If you don’t have variables available at inference time, don’t panic. These features can still be very useful for your forecast. For example: details on inventory, and shipments/supply can be leading predictors even if future details are unknown. Unavailable at forecast examples:

* Future promotions
* Future prices
* Future holidays
* Future inventory

<img src='imgs/unavailable_at_forecast_covariates.png'>

