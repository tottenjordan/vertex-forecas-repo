# Intuition behind past-only covariates

<img src='imgs/intuition_past_only_covariates.png'>

## When recommending tracks to users, deep learning models cluster similar artists/tracks by similarity

<img src='imgs/recsys_analogy_v2.png'>

## We can apply this same concept to deep learning forecast models: they cluster time series by their similarity (e.g., demand patterns, feature representations, etc.)
<img src='imgs/applied_to_dl_forecast_v2.png'>

## So, time series with similar profiles are closer in the embedding space
<img src='imgs/ts_w_similar_profiles_v1_v2.png'>

## But what if the *similar profiles* don’t capture the whole story?
*In this case, adding a feature for inventory reveals these two time series could be dropping for different reasons*
<img src='imgs/ts_w_similar_profiles_v2_v2.png'>

## We create better forecast models with feature sets that better capture reality. 
*In this case, adding a feature for inventory reveals these two time series could be dropping for different reasons*

<img src='imgs/ts_w_similar_profiles_v3_v2.png'>