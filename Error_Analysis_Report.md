# Regression Model Error Analysis

## Executive Summary

The Ridge regression model achieves a **MAE of $133,513** and **RMSE of $907,821** on the chronological test set.

The large gap between MAE and RMSE is mainly caused by a small number of extreme prediction errors, particularly a **$26.59M Kent property** that was predicted at only **$210.5K**.

Model error is relatively stable across low, medium, and high price bands but increases substantially for **very-high-priced properties**, suggesting difficulty modeling the upper end of the market.

City-level results should be interpreted alongside sample size. **Seattle and Bellevue** have the largest samples, while cities with only 2–5 observations provide limited evidence.

The waterfront subgroup contains only **5 observations**, so no reliable conclusion can be drawn about waterfront performance.

Finally, all test observations are from **2014**. Although the split is chronological, the evaluation does not measure multi-year temporal generalization.

## Overall Performance

| Metric | Value |
|---|---:|
| MAE | $133,512.74 |
| RMSE | $907,820.70 |

- **MAE** measures the average absolute prediction error.
- **RMSE** penalizes large errors more heavily.
- The much higher RMSE indicates that a few extreme errors have a disproportionate impact.

## Error by Price Band

| Price Band | Observations | Mean Absolute Error | Median Absolute Error |
|---|---:|---:|---:|
| Low | 218 | $78,006 | $62,857 |
| Medium | 218 | $82,763 | $59,101 |
| High | 219 | $95,267 | $69,157 |
| Very High | 216 | $279,531 | $98,891 |

Error is relatively consistent across the low, medium, and high bands. The **very-high band is substantially less accurate**, with MAE increasing to approximately **$280K**.

## Error by City

| City | Observations | Mean Absolute Error | Median Absolute Error |
|---|---:|---:|---:|
| Kent | 41 | $709,740 | $67,919 |
| Clyde Hill | 3 | $508,341 | $90,960 |
| Medina | 3 | $317,855 | $228,078 |
| Mercer Island | 20 | $238,048 | $179,518 |
| Normandy Park | 4 | $175,235 | $160,669 |
| Vashon | 5 | $154,629 | $62,940 |
| Skykomish | 2 | $149,151 | $149,151 |
| Bellevue | 54 | $139,595 | $100,680 |
| Seattle | 295 | $122,461 | $76,581 |
| Snoqualmie | 11 | $120,508 | $75,773 |

Kent's high MAE is strongly influenced by the **$26.59M outlier** discussed below.

Results for cities with only 2–5 observations should be treated as **directional rather than representative**. Seattle and Bellevue provide more reliable estimates due to their larger sample sizes.

## Error by Waterfront Status

| Waterfront | Observations | Mean Absolute Error | Median Absolute Error |
|---|---:|---:|---:|
| No | 866 | $132,791 | $69,909 |
| Yes | 5 | $258,587 | $332,088 |

Waterfront properties show higher observed errors, but with only **5 observations**, the sample is too small to conclude that the model systematically performs worse on waterfront properties.

## Error by Sale Year

| Sale Year | Observations | Mean Absolute Error | Median Absolute Error |
|---|---:|---:|---:|
| 2014 | 871 | $133,513 | $69,950 |

All test observations are from **2014**. The chronological split avoids future-to-training leakage, but the test set does not provide a genuine **multi-year temporal holdout**.

Therefore, these results demonstrate performance on later observations in the dataset rather than robust generalization to future market conditions.

## Largest Prediction Errors

| Actual Price | Prediction | Absolute Error | City | ZIP | Living Area | Waterfront |
|---:|---:|---:|---|---|---:|---|
| $26,590,000 | $210,512 | $26,379,488 | Kent | 98031 | 1,180 sqft | No |
| $3,800,000 | $2,373,247 | $1,426,753 | Clyde Hill | 98004 | 7,050 sqft | No |
| $2,300,000 | $1,518,133 | $781,867 | Seattle | 98119 | 3,970 sqft | No |
| $2,351,956 | $1,595,861 | $756,095 | Mercer Island | 98040 | 5,010 sqft | No |
| $1,755,000 | $1,019,020 | $735,980 | Seattle | 98112 | 2,360 sqft | No |

## Extreme $26.59M Observation

The largest error is a **Kent property sold for $26.59M**, while the model predicts approximately **$210.5K**.

Key characteristics:

- 3 bedrooms
- 2 bathrooms
- 1,180 sqft
- No waterfront designation
- No recorded view
- Kent, WA 98031

The unusually high sale price relative to the property's recorded characteristics makes this a **potential data-quality outlier** and warrants verification against the source data.

Because RMSE squares errors, this single observation has a major impact on the overall RMSE. However, it **should not be removed solely to improve model metrics**. Any correction or exclusion should follow an explicit, documented data-quality rule.

## Key Findings & Limitations

- **Typical error is much lower than RMSE suggests:** MAE is ~$134K compared with RMSE of ~$908K.
- **Upper-price performance is weaker:** The very-high price band has an MAE of ~$280K.
- **The $26.59M Kent property dominates extreme-error analysis** and should be verified.
- **Small city samples limit subgroup conclusions**, especially for cities with 2–5 observations.
- **Waterfront performance cannot be reliably assessed** because only 5 waterfront properties are in the test set.
- **Multi-year temporal generalization remains untested** because all test observations are from 2014.