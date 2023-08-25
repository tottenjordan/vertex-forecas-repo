# Probabilistic Inference with Vertex Forecast

see the [Probabilistic Inference User Guide](https://docs.google.com/document/d/1kegOsor8j7HO2qttMKK6mtfl5GzoxDf8LhsXH8oXsyo/edit#heading=h.pkq5rspaeaz9) for more details

## Intermittent demand

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/knowledge-share/imgs/1_trouble_w_point_frcsts.png)

## Dealing with Uncertainty

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/knowledge-share/imgs/2_use_probabilistic.png)

## Vertex Forecast

![alt text](https://github.com/tottenjordan/vertex-forecas-repo/blob/main/knowledge-share/imgs/4_use_probabilistic.png)


## Practical use

[1] `Target Service Level` 

* quantity should be determined by the business:
> * profitability, 
> * supply chain capacity 
> * store space, 
> * branding / marketing, etc.

* ideally do this in a purely data driven manner e.g., ABC/XYZ analysis

* two ways to do this:
> * Use the quantiles to calculate the required safety stock, and add it to the base ROQ
> * Calculate an optimal ROQ directly from the target service level and the forecasted quantile