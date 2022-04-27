# Wearable Data Privacy
Investigating the privacy inherent to publicly available, anonymized datasets of wearable data (mainly time-series and daily/hourly physiological data, such as activity, EKG, etc.).  Exploration of differentially privacy solutions.

- This repository evaluates the relative entropy (e.g. amount of information contained within any given field), uniqueness, and privacy risk of the raw anonymized dataset.  Afterwards, a (Euclidean) distance-based attack is created and functionalized, allowing for multi-trial attacks against a dataset (currently sampling 5% of the data for testing).  Finally, a number of privacy mechanisms are implemented and then evaluated, using the distance-based attack previously described.

## Datasets
1. https://www.kaggle.com/datasets/arashnic/fitbit
2. https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/TGGCSZ (unused but potentially could be)

## Limitations
1. Small dataset (31 users)
2. K-anonymity may be more powerful in these circumstances

## References
1. https://www.microsoft.com/en-us/research/publication/collecting-telemetry-data-privately/
