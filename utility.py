import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import datetime as dt
from math import ceil
from scipy.stats import boxcox


def simple_fillna(df):
    """Takes in a pandas dataframe and fills the missing values with the mean
    or the mode, depending on the data type of each column.

    Arguments:
        df {pandas.DataFrame} -- Arbitrary pandas.DataFrame with missing
        values.

    Returns:
        pandas.DataFrame -- Same dataframe, but with all np.nans replaced by
        either the mean or the mode of each column.
    """
    df.dropna(axis=1, how='all', inplace=True)
    for feature in df.columns:
        if df[feature].dtype != 'object':
            df[feature] = df[feature].fillna(df[feature].mean())
        elif df[feature].dtype == 'object':
            df[feature] = df[feature].fillna(df[feature].mode()[0])

    return df


def normalize_df(df: pd.DataFrame, col_to_dummy: str, col_to_time: list,
                 date_format='%Y-%m-%d') -> pd.DataFrame:
    """Normalizes pandas.DataFrame

    Arguments:
        df {pandas.DataFrame} -- DataFrame to be normalized.
        col_to_dummy {str} -- List-like column to be turned into dummys.
        col_to_time {str} -- Column to turn into timeperiod.

    Keyword Arguments:
        date_format {str} -- Date format found in column
        (default: {'%Y-%m-%d'}).

    Returns:
        pandas.DataFrame -- Normalized DataFrame
    """

    objects = [obj for obj in df.columns if df[obj].dtype == 'object']

    new_df = _amenities_to_list(df, col_to_dummy)
    new_df = _normalize_datetime(new_df, date_format, col_to_time)
    new_df = _normalize_string(new_df, objects)

    return new_df


def _amenities_to_list(df, col_name):
    extract_amenities = re.compile(r'[^{}",]+')
    df.loc[:, col_name] = df.loc[:, col_name].map(
        lambda x: re.findall(extract_amenities, str(x)))
    amenities_list = []
    for sub_list in df.loc[:, col_name].tolist():
        amenities_list += sub_list
    amenities_list = list(set(amenities_list))
    for amenity in amenities_list:
        df['has_'+amenity] = df.loc[:, col_name].map(lambda x: amenity in x)

    return df


def _normalize_datetime(df, format, col_to_time):
    new_df = df.copy()
    for feature in col_to_time:
        date_series = new_df[feature].dropna().map(
            lambda x: dt.datetime.strptime(x, format))
        max_date = date_series.max()
        days_series = date_series.map(lambda x: (max_date-x).days)
        new_df.loc[new_df[feature].notnull(), feature] = days_series

    return new_df


def _normalize_string(df, objects):
    sub_others = re.compile(r'[^a-zA-Z0-9\.]')

    new_df = df

    for obj in objects:
        new_series = new_df[obj].map(lambda x: re.sub(sub_others, '', str(x)))
        new_series = new_series.map(_reinst_nan)
        new_series = pd.to_numeric(new_series, errors='ignore')

        new_df[obj] = new_series

    return new_df


def _reinst_nan(string):
    if string == 'nan':
        return np.nan
    else:
        return string


def num_feature_selection(df_num, target: str, max_degree=1) -> dict:
    """Measures correlation between each feature in df and the target up
    to max_degree. Returns dictionary with highest correlation and degree of
    highest correlation.

    Arguments:
        df_num {pandas.DataFrame} -- DataFrame containing the target and
        numerical variables to be tested.
        target {str} -- Target of correlation test.

    Keyword Arguments:
        max_degree {int} -- Maximum degree to which to calculate correlation
         (default: {1}).

    Returns:
        dict -- Dict containing the degree of highest correlation and the
        according correlation.
    """
    corr_dict = {}
    features = [feature for feature in df_num.columns if feature != target]

    for feature in features:
        corr_dict[feature] = (0, 0)
        for degree in range(max_degree + 1):
            corr = df_num[target].corr((df_num[feature] ** degree))
            if abs(corr) > abs(corr_dict[feature][1]):
                corr_dict[feature] = (degree, corr)

    return corr_dict


def num_visual(corr_dict: dict, df, target: str, threshold=0.25,
               figsize=(10, 10), filename=None) -> None:
    """Shows seaborn's regplots for features in dictionary and dataframe

    Arguments:
        corr_dict {dict} -- Dict from function num_feature_selection.
        df {pandas.DataFrame} -- DataFrame with which corr_dict was calculated.
        target {str} -- Target column of analysis.

    Keyword Arguments:
        threshold {float} -- Minimum correlation to be visualized
        (default: {0.25}).
        figsize {tuple} -- Size of subplots in visulization
        (default: {(10, 10)}).
        filename {str or None} -- File to save the regplots to
        (default: {None}).

    Returns:
        None
    """
    chosen_features = [feature for feature in corr_dict.keys() if abs(
        corr_dict[feature][1]) >= threshold]

    fig, ax = plt.subplots(ceil(len(chosen_features) / 3,), 3, figsize=figsize)

    row = 0
    col = 0
    for feature in chosen_features:
        sns.regplot(x=feature, y=target, data=df, ax=ax[row][col],
                    order=corr_dict[
                                    feature
                                    ][0]
                    ).set_title('{} vs. {}'.format(
                                                   feature,
                                                   target
                                                                   ))
        plt.tight_layout()
        if col == 2:
            col = 0
            row += 1
        elif col < 2:
            col += 1
    if filename:
        plt.savefig(filename)

    plt.show()

    return None


def dist_visualizer(df, figsize=(10, 10), filename=None) -> None:
    """Visualized distribution of numerical variables in pandas DataFrame via
    seaborn's distplot.

    Arguments:
        df {pandas.DataFrame} -- DataFrame containing the variables to be
        visualized.

    Keyword Arguments:
        figsize {tuple} -- Figure size of subplots (default: {(10, 10)})
        filename {str or None} -- File to store the plot in (default: {None}).

    Returns:
        None
    """
    rows = int(ceil(df.shape[1] / 2))
    cols = 2
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    row = 0
    col = 0
    for feature in df.columns:
        sns.distplot(df.loc[:, feature].dropna().astype(
            'float64'), ax=ax[row][col]).set_title(feature)
        if col:
            col = 0
            row += 1
        else:
            col += 1
        plt.tight_layout()
    if filename:
        plt.savefig(filename)

    return None


def adjust_distribution(df: pd.DataFrame, selected_columns: list,
                        figsize=(10, 10), filename=None) -> pd.DataFrame:
    """Adjusts distributions in DataFrame for selected columns.

    Arguments:
        df {pd.DataFrame} -- DataFrame containing the data to be adjusted
        selected_columns {list} -- Columns to be adjusted

    Keyword Arguments:
        figsize {tuple} -- Size of subplots (default: {(10, 10)}).
        filename {[type]} -- File to save plot to (default: {None}).

    Returns:
        pd.DataFrame -- DataFrame containing adjusted data.
    """
    for feature in selected_columns:
        transform_array = np.array(df.loc[np.logical_and(
            df[feature] > 0, df[feature].notnull()), feature])
        transformed_array, _ = boxcox(transform_array, None)
        df.loc[np.logical_and(
            df[feature] > 0,
            df[feature].notnull()), feature] = transformed_array
    dist_visualizer(df.loc[:, selected_columns])
    if filename:
        plt.savefig(filename)
    return df


def obj_selector(df_cat: pd.DataFrame, target: str, max_cats=10,
                 ignore_max_cat=[]) -> dict:
    """Shows correlation between dummys of each feature in df and the target
    as a dict.

    Arguments:
        df_cat {pd.DataFrame} -- DataFrame containing the features and the
        target.
        target {str} -- Target column.

    Keyword Arguments:
        max_cats {int} -- Maximum categories to allow a feature to have.
        Features with more distinct categories are excluded from the analysis
        (default: {10}).
        ignore_max_cat {list} -- List of features,w chich are are
        allowed to ignore max_cats (default: {[]}).

    Returns:
        dict -- Dictionary containing each features' dummys maximum and mean
        correlation to the target.
    """
    features_dict = {}
    for col in df_cat.columns:
        if (
            (df_cat[col].nunique() <= max_cats or col in ignore_max_cat) and
                (df_cat[col].nunique() > 1) and
                (col != target)):
            sub_df = df_cat.loc[df_cat[target].notnull(), [col, target]]
            dummy_df = pd.get_dummies(sub_df,
                                      dummy_na=True,
                                      columns=[col],
                                      drop_first=False)
            corr_list = [abs(dummy_df[target].corr(dummy_df[cat]))
                         for cat in dummy_df.columns if cat != target]
            features_dict[col] = (sum(corr_list) / len(corr_list),
                                  max(corr_list))

    return features_dict


def obj_visualizer(feature_dict: dict, df: pd.DataFrame, target: str,
                   threshold=0.1, figsize=(10, 10),
                   filename=None) -> None:
    """Visualizes relationship between features in feature_dict and the target.
    Uses seaborn's boxplot.

    Arguments:
        feature_dict {dict} -- Dictionary from obj_selector.
        df {pd.DataFrame} -- DataFrame conatining the data to be visualized.
        target {str} -- Target column in df.

    Keyword Arguments:
        threshold {float} -- Minimum required maximum correlation of a feature
        to be displayed (default: {0.1}).
        figsize {tuple} -- Size of subplots (default: {(10, 10)}).
        filename {[type]} -- File to save plot in (default: {None}).

    Returns:
        None
    """
    selected_features = [feature for feature in feature_dict.keys() if abs(
        feature_dict[feature][1]) >= threshold]
    rows, cols = int(ceil(len(selected_features)/2)), 2
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    row, col = 0, 0
    for feature in selected_features:
        if col != target:
            sns.boxplot(x=feature, y=target, data=df,
                        ax=ax[row][col]).set_title(feature)
            ax[row][col].set_ylim(df[target].mean(
            ) - df[target].std() * 4, df[target].mean() + df[target].std() * 4)
            plt.setp(ax[row][col].xaxis.get_majorticklabels(), rotation=45)
            if col:
                col = 0
                row += 1
            else:
                col += 1
        plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

    return None


def concat_to_single(row_in_df: pd.Series) -> str:
    """Concatenates strings in pandas.Series to single string.

    Arguments:
        row_in_df {pd.Series} -- Row containing strings to concatenate.

    Returns:
        str -- Concatenated string.
    """
    new_string = ''
    for col in row_in_df:
        if str(col) != 'nan':
            new_string = new_string + str(col)
    return new_string


def score_binning(score_cell: float) -> str:
    """Turns numeric values in category strings.

    Arguments:
        score_cell {float} -- Number to turn into bin.

    Returns:
        str -- Category.
    """
    if score_cell <= 100 and score_cell > 80:
        return '3'
    elif score_cell <= 80 and score_cell > 60:
        return '2'
    elif score_cell <= 60 and score_cell > 40:
        return '1'
    else:
        return '0'
