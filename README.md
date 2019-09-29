# Airbnb Listings Exploration (Seattle and Boston)

The code at hand was used to get a deeper look into the relationships and
connections in Airbnb listing data. In particular, the coda was used to get
an insight into which features of the individual listings were most the
most explanative of the price. Furthermore, in an exploration of NLP
possibilities of the data, the review scores were grouped in bins, which were
then predicted by a GradientBoostingClassifier, using only the various
descriptions of the objects authored by the hosts.

## Getting Started

To get the notebooks, the data, as well as the utility functions on your local
machine, you can clone the repo via:

```bash
git clone https://github.com/MarcoCaglia/blogpost_udacity.git
```

### Prerequisites

Find below the libraries used in the notebook:

1. pandas 0.25.1
2. numpy 1.15.5
3. matplotlib 3.3.1
4. seaborn 0.9.0
5. scikit-learn 0.21.2
6. imbalanced-learn 0.5.0

```bash
conda install -c conda-forge imbalanced-learn=0.5.0
```

The python version was Python 3.6.9.

### Files in the Repository

1. listings_boston.csv
   + This csv file contains the data of Airbnb listings, which will be used
      in the analysis. It contains only the Boston data.
2. listings_seattle.csv
    + Analogous to listings_boston.csv. Contains only Seattle data.
3. price_determinants_preprocessing.ipynb
   + Concatenation and following preprocessing of the Seattle and Boston data.
4. price_determinants_ml_rdy.csv
   + Output of price_determinants_preprocessing.ipynb. Preprocessed input for
     price_determinants_modelling.ipynb.
5. price_determinants_modelling.ipynb
   + linear regression of the listings' price on the listings' characteristics
     and summary of results.
6. text_inference.ipynb
   + Notebook for text analysis of Airbnb listings. Contains
     GradientBoostingClassifier trained on vectorized and concatenated
     descriptions. Does not require preprocessed input.
7. utility.py
   + Contains all custom functions used in the notebooks, including
     documentation.

### Summary of Results

The analysis of the price determinants of Airbnb listings has shown,
that only a few features lead to an explanation of the variance of the price
of 62.5 per cent. Some features have a noteably greater influence on the price
than others. Especially features regarding the room type and the size proof to
be rather influential.

The textual analysis has shown, that it is possible to group the review_ratings
into bins and predict those bins, using only the textual descriptions of the
listing. The model contained in the data reaches a weighted F1-Score of 69.7
per cent on a test set.
