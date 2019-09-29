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

1.pandas 0.25.1
2.numpy 1.15.5
3.matplotlib 3.3.1
4.seaborn 0.9.0
5.scikit-learn 0.21.2
6.imbalanced-learn 0.5.0

```bash
conda install -c conda-forge imbalanced-learn=0.5.0
```

The python version was Python 3.6.9.
