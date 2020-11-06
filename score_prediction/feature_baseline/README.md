## Features baseline

### Mean score baseline
The simplest baseline is computed by outputting the average score for all query images:  
`python features_baseline.py`   
`--output_mean_score 1`   
`--data_root /path/to/theDataset/cosmology_aux_data_170429/cosmology_aux_data_170429/`   

### Feature based baseline

The features are fractions of total energy in each coefficient in Coiflet Discrete Wavelet Transform (DWT).  
[Histogram-based Gradient Boosting Regression Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
is fitted on the features.  
`python features_baseline.py`   
`--data_root /path/to/theDataset/cosmology_aux_data_170429/cosmology_aux_data_170429/`   

To reproduce the features please run `Mathematica_feature_extraction.nb` script which computes the [DWT](https://reference.wolfram.com/language/ref/DiscreteWaveletTransform.html).
