# Probabilistic Inference with Vertex Forecast

see the [Probabilistic Inference User Guide](https://docs.google.com/document/d/1kegOsor8j7HO2qttMKK6mtfl5GzoxDf8LhsXH8oXsyo/edit#heading=h.pkq5rspaeaz9) for more details

### Dealing with Uncertainty

**Retail sales are random by nature**; consider the following scenario:

* TODO

#### business value

**a forecast of (1) a distribution of values vs (2) a single predicted value, has the following advantages:**

* **Make better decisions:** Balance risks and costs, such as missed sales from understocking vs. the shipping + holding costs of overstocking.
* **Provide better explanations:** Understand the range of possible outcomes, such as the likelihood of low sales vs. high sales if a product is trending rather than a single point prediction for median sales.
* **Build more trust:** Low uncertainty for accurate predictions and high uncertainty when the model has insufficient information helps supply chain planners develop trust and act accordingly.

#### Probabilistic Inference

The premise of **Probabilistic Inference** is to learn a predictive distribution during training, and infer statistics of the distribution such as the **mean** and **quantiles** (including **median**) during prediction. Advantages over point predictions include:

* **Quantifies Uncertainty:** Includes quantiles of the predictive distribution, which describes the range of possible outcomes, and expresses the confidence of the prediction.

* **Models Sparsity:** Explicitly models the likelihood of zero sales (ex. stockout) and counts of events (ex. sales) as often occur in Retail demand forecasting, which may significantly improve the accuracy of predictions for products with low volume sales (sparsity).

* **Learns Adaptively:** Places the greatest emphasis on learning where the model is the most  confident, avoiding overfitting to noise in the data. This also places an even emphasis across scales of the target, enabling the model to make accurate predictions for both slow and fast selling items. Together, this can significantly reduce bias in forecasts.

* **Automates Distribution Fit:** Fits the predictive distribution using a number of candidate distributions which describe different types of processes, and weights according to the best fit for the use case, requiring no additional input from the user.


#### practical use

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