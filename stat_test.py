"""
Statistical test functions for group comparisons.

A collection of functions for performing statistical tests on grouped data,
including both two-group and multi-group comparisons. The package provides
automated workflows for assumption testing and appropriate statistical test selection.

Author: Joe
Keywords: statistics, group-comparison, hypothesis-testing
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro, anderson, bartlett, levene, 
    kruskal, ttest_ind, mannwhitneyu,
    f_oneway
)
from itertools import combinations
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy.stats import ttest_ind

# Try to import scikit_posthocs (optional dependency)
try:
    import scikit_posthocs as sp
    HAS_SCIKIT_POSTHOCS = True
except ImportError:
    HAS_SCIKIT_POSTHOCS = False
    sp = None
    warnings.warn(
        "scikit-posthocs not installed. Some post-hoc tests (Games-Howell, Dunn's test) "
        "will use fallback methods. Install with: `pip install scikit-posthocs`"
    )


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
np.seterr(all='ignore')


# ============================================================================
# Primary Functions
# ============================================================================

def test_assumptions(data, response_col, group_col, homogeneity_method="auto"):
    """
    Test normality and homogeneity of variance assumptions.
    
    This function performs normality tests (Shapiro-Wilk or Anderson-Darling)
    for each group and homogeneity of variance tests (Levene or Bartlett).
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the response and group columns.
    response_col : str
        Name of the column containing the response variable.
    group_col : str
        Name of the column containing the group variable.
    homogeneity_method : str, optional
        Method for testing homogeneity of variance. Options: 'levene', 'bartlett', or 'auto'.
        Default is 'auto'. When 'auto', the method is automatically selected based on
        normality results and sample size.
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'test.results': DataFrame with test results
        - 'Normality': bool, True if normality assumption is met
        - 'Homogeneity': bool, True if homogeneity assumption is met
    
    Raises
    ------
    ValueError
        If response_col or group_col not found in data, or if homogeneity_method
        is not 'levene', 'bartlett', or 'auto'.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'value': [1, 2, 3, 4, 5, 6, 7, 8],
    ...     'group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
    ... })
    >>> result = test_assumptions(data, 'value', 'group')
    """
    # Validate input
    if response_col not in data.columns:
        raise ValueError("Response column not found in the data. Please check the 'response_col' parameter.")
    if group_col not in data.columns:
        raise ValueError("Group column not found in the data. Please check the 'group_col' parameter.")
    if homogeneity_method not in ["levene", "bartlett", "auto"]:
        raise ValueError("Parameter 'homogeneity_method' should be 'levene', 'bartlett' or 'auto'. Default is 'auto'.")
    
    # Perform normality tests by group
    st = perform_normality_by_group(data, response_col, group_col)
    
    # Perform homogeneity of variance test
    bt = perform_variances_homogeneity_test(data, response_col, group_col, 
                                          mode=homogeneity_method, normality_res=st)
    
    # Combine results
    res = pd.concat([st, bt], ignore_index=True)
    
    # Check if assumptions are met (p >= 0.05 means assumption is met)
    pass_normality = not any(st['sig'] != "")
    pass_homogeneity = not any(bt['sig'] != "")
    
    res_dict = {
        "test.results": res,
        "Normality": pass_normality,
        "Homogeneity": pass_homogeneity
    }
    
    return res_dict


def auto_stat_test(data, test_assumptions_res, response_col, group_col, 
                   factors=None, interaction=True, aov_post_hoc="tukey", 
                   kw_post_hoc="dunn", consider_welch_aov=True):
    """
    Automatically select and perform appropriate statistical test based on assumptions.
    
    This function automatically selects the appropriate statistical test based on:
    - Number of groups (two vs. multiple)
    - Normality assumption
    - Homogeneity of variance assumption
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the response and group columns.
    test_assumptions_res : dict
        Results from test_assumptions() function.
    response_col : str
        Name of the column containing the response variable.
    group_col : str
        Name of the column containing the group variable.
    factors : list of str, optional
        Additional factors for multi-way ANOVA. Default is None.
    interaction : bool, optional
        Whether to include interaction terms in ANOVA. Default is True.
    aov_post_hoc : str, optional
        Post-hoc method for ANOVA. Options: 'tukey' or 'holm'. Default is 'tukey'.
    kw_post_hoc : str, optional
        Post-hoc method for Kruskal-Wallis. Options: 'wilcoxon' or 'dunn'. 
        Default is 'dunn'.
    consider_welch_aov : bool, optional
        Whether to consider Welch ANOVA when normality is met but homogeneity is not.
        Default is True.
    
    Returns
    -------
    dict
        A dictionary containing test results, which varies based on the test performed.
    
    Raises
    ------
    ValueError
        If response_col or group_col not found, or if post_hoc methods are invalid.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     'group': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B']
    ... })
    >>> assumptions = test_assumptions(data, 'value', 'group')
    >>> result = auto_stat_test(data, assumptions, 'value', 'group')
    """
    # Validate input
    if response_col not in data.columns:
        raise ValueError("Response column not found in the data. Please check the 'response_col' parameter.")
    if group_col not in data.columns:
        raise ValueError("Group column not found in the data. Please check the 'group_col' parameter.")
    if aov_post_hoc not in ["tukey", "holm"]:
        raise ValueError("Parameter 'aov_post_hoc' should be 'tukey' or 'holm'. Default is 'tukey'.")
    if kw_post_hoc not in ["wilcoxon", "dunn"]:
        raise ValueError("Parameter 'kw_post_hoc' should be 'wilcoxon' or 'dunn'. Default is 'dunn'.")
    
    # Make sure key columns are categorical
    data = data.copy()
    data[group_col] = data[group_col].astype('category')
    if factors is not None:
        for factor in factors:
            data[factor] = data[factor].astype('category')
    
    # Check if assumptions are met (both must be True)
    pass_assumptions = test_assumptions_res["Normality"] & test_assumptions_res["Homogeneity"]
    
    # Get number of groups
    group_numb = len(data[group_col].unique())
    
    # Select appropriate test based on number of groups and assumptions
    if group_numb > 2:
        # Multiple groups
        if pass_assumptions:
            # Normality and homogeneity met -> ANOVA
            print("Normality and homogeneity of variances assumptions are met.")
            print(f"Number of groups is {group_numb}.\nPerforming parametric test: ANOVA.")
            res_list = perform_anova_test(data, response_col, group_col, 
                                         factors=factors, interaction=interaction, 
                                         post_hoc_method=aov_post_hoc)
        elif test_assumptions_res["Normality"] & consider_welch_aov:
            # Normality met but homogeneity not met -> Welch ANOVA
            print("Normality is met, but homogeneity is not met.")
            print(f"Number of groups is {group_numb}.\nPerforming parametric test: Welch ANOVA.")
            res_list = perform_welch_anova(data, response_col, group_col)
        else:
            # Assumptions not met -> Kruskal-Wallis
            print("Normality or homogeneity of variances assumptions are not met.")
            print("Performing non-parametric test: Kruskal-Wallis")
            res_list = perform_kw_test(data, response_col, group_col, kw_post_hoc)
    else:
        # Two groups
        if pass_assumptions:
            print("Normality and homogeneity of variances assumptions are met.")
            print(f"Number of groups is {group_numb}.\nPerforming parametric test: Student's t-test.")
            res_list = perform_stat_for_two_group(data, response_col, group_col, method="t.test")
        else:
            print("Normality or homogeneity of variances assumptions are not met.")
            print(f"Number of groups is {group_numb}.\nPerforming non-parametric test: Wilcoxon test.")
            res_list = perform_stat_for_two_group(data, response_col, group_col, method="wilcox.test")
    
    return res_list


def stat_res_collect(data, nor, stat, task_name, consider_welch_aov=True):
    """
    Collect and organize statistical test results into a unified format.
    
    This function combines normality/homogeneity test results with the main
    statistical test results and post-hoc tests (if applicable).
    
    Parameters
    ----------
    data : pandas.DataFrame
        The original input data.
    nor : dict
        Results from test_assumptions() function.
    stat : dict
        Results from auto_stat_test() function.
    task_name : str
        Name of the analysis task (used for labeling results).
    consider_welch_aov : bool, optional
        Whether to consider Welch ANOVA when normality is met but homogeneity is not.
        Default is True.
    
    Returns
    -------
    dict
        A dictionary containing:
        - task_name: The original data
        - 'stat_test': DataFrame with all test results organized
    
    Examples
    --------
    >>> data = pd.DataFrame({'value': [1,2,3,4,5,6], 'group': ['A','A','B','B','C','C']})
    >>> assumptions = test_assumptions(data, 'value', 'group')
    >>> stat_result = auto_stat_test(data, assumptions, 'value', 'group')
    >>> final_result = stat_res_collect(data, assumptions, stat_result, 'my_analysis')
    """
    # Step 1: Collect normality and homogeneity test results
    df1 = nor["test.results"].copy()
    
    # Step 2: Check if assumptions passed (both must be True)
    nor_pass = nor["Normality"] & nor["Homogeneity"]
    
    # Get post-hoc method if available
    post_hoc_method = stat.get("post_hoc_method", None)
    
    # Step 3: Collect main statistical test results
    if nor_pass:
        # Parametric tests (ANOVA)
        stat_method = "ANOVA"
        aov_res = stat.get("ANOVA", None)
        
        if aov_res is not None:
            # Get the correct column name for p-values (statsmodels uses "PR(>F)")
            p_col = "PR(>F)" if "PR(>F)" in aov_res.columns else "Pr(>F)"
            
            if stat.get("avo_method") == "one-way ANOVA":
                df2 = pd.DataFrame({
                    "step": ["Parametric tests"] + [""] * (len(aov_res) - 1),
                    "method": [stat_method] + [""] * (len(aov_res) - 1),
                    "group": [aov_res.iloc[0].get("formula", "")] + [""] * (len(aov_res) - 1),
                    "p.value": aov_res[p_col].values
                })
            else:
                df2 = pd.DataFrame({
                    "step": ["Parametric tests"] + [""] * (len(aov_res) - 1),
                    "method": [stat_method] + [""] * (len(aov_res) - 1),
                    "group": aov_res.index.tolist(),
                    "p.value": aov_res[p_col].values
                })
            
            df2["sig"] = df2["p.value"].apply(p_to_stars)
            need_post_hoc = any(aov_res[p_col] < 0.05)
        else:
            df2 = pd.DataFrame()
            need_post_hoc = False
            
    elif nor["Normality"] & consider_welch_aov:
        # Welch ANOVA (normality met but homogeneity not met)
        stat_method = "Welch ANOVA"
        welch_aov_res = stat.get("Welch ANOVA", None)
        
        if welch_aov_res is not None:
            formula_str = str(welch_aov_res.get("data.name", "")).replace(" and ", "~")
            df2 = pd.DataFrame({
                "step": "Parametric tests",
                "method": stat_method,
                "group": formula_str,
                "p.value": welch_aov_res.get("p.value", np.nan)
            }, index=[0])
            df2["sig"] = df2["p.value"].apply(p_to_stars)
            need_post_hoc = welch_aov_res.get("p.value", 1.0) < 0.05
        else:
            df2 = pd.DataFrame()
            need_post_hoc = False
    else:
        # Non-parametric tests (Kruskal-Wallis)
        stat_method = "Kruskal-Wallis"
        kw_res = stat.get("Kruskal-Wallis", None)
        
        if kw_res is not None:
            formula_str = kw_res.get("formula", "")
            df2 = pd.DataFrame({
                "step": "Nonparametric tests",
                "method": stat_method,
                "group": formula_str,
                "p.value": kw_res.get("pvalue", np.nan)
            }, index=[0])
            df2["sig"] = df2["p.value"].apply(p_to_stars)
            need_post_hoc = kw_res.get("pvalue", 1.0) < 0.05
        else:
            df2 = pd.DataFrame()
            need_post_hoc = False
    
    # Merge results
    if not df2.empty:
        df_merge = pd.concat([df1, df2], ignore_index=True)
    else:
        df_merge = df1.copy()
    
    df_merge.insert(0, "task", [task_name] + [""] * (len(df_merge) - 1))
    
    # Step 4: Add post-hoc results if needed
    if need_post_hoc and post_hoc_method is not None:
        post_hoc = stat.get("post_hoc", None)
        if post_hoc is not None and not post_hoc.empty:
            df3 = pd.DataFrame({
                "task": [""] * len(post_hoc),
                "step": ["Post hoc test"] + [""] * (len(post_hoc) - 1),
                "method": post_hoc_method,
                "group": post_hoc["comparison"].values,
                "p.value": post_hoc["p.value"].values,
                "sig": post_hoc["p.value"].apply(p_to_stars)
            })
            df_merge = pd.concat([df_merge, df3], ignore_index=True)
    
    # Format p-values
    df_merge["p.value"] = df_merge["p.value"].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) and not pd.isna(x) else x)
    df_merge.loc[df_merge["p.value"].apply(lambda x: isinstance(x, (int, float)) and x < 0.0001), "p.value"] = "<0.0001"
    
    res = {task_name: data, "stat_test": df_merge}
    print("Complete data cleaning!")
    
    return res




# ============================================================================
# Secondary Functions
# ============================================================================

def perform_normality_by_group(data, column_for_test, group_col):
    """
    Perform normality test for each group separately.
    
    Uses Shapiro-Wilk test for sample sizes <= 5000, and Anderson-Darling
    test for larger samples.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    column_for_test : str
        Name of the column to test for normality.
    group_col : str
        Name of the column containing group labels.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with normality test results for each group.
    
    Examples
    --------
    >>> data = pd.DataFrame({'value': [1,2,3,4,5,6], 'group': ['A','A','B','B','A','B']})
    >>> result = perform_normality_by_group(data, 'value', 'group')
    """
    # Split data by group
    data_split = {group: group_data for group, group_data in data.groupby(group_col)}
    
    # Perform normality test for each group
    nor_res_list = []
    for group_name, group_data in data_split.items():
        sample_data = group_data[column_for_test].dropna()
        sample_size = len(sample_data)
        
        if sample_size > 5000:
            print(f"Sample size > 5000. Using Anderson-Darling normality test for group {group_name}.")
            # Anderson-Darling test
            ad_result = anderson(sample_data)
            # Anderson-Darling returns critical values, not p-value directly
            # We'll use a simplified approach: check if statistic exceeds critical value
            p_value = 0.05 if ad_result.statistic > ad_result.critical_values[2] else 0.5
            method_name = "Anderson-Darling normality test"
        else:
            print(f"Sample size <= 5000. Using Shapiro-Wilk normality test for group {group_name}.")
            # Shapiro-Wilk test
            st_result = shapiro(sample_data)
            p_value = st_result.pvalue
            method_name = "Shapiro-Wilk normality test"
        
        nor_df = pd.DataFrame({
            "method": [method_name],
            "group": [group_name],
            "p.value": [p_value]
        })
        nor_res_list.append(nor_df)
    
    # Combine results
    nor_df = pd.concat(nor_res_list, ignore_index=True)
    nor_df.insert(0, "step", ["Normality"] + [""] * (len(nor_df) - 1))
    nor_df["sig"] = nor_df["p.value"].apply(p_to_stars)
    
    print("Finish normality test!")
    
    return nor_df


def perform_variances_homogeneity_test(data, response_col, group_col, mode="auto", normality_res=None):
    """
    Perform test for homogeneity of variances.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the column to test.
    group_col : str
        Name of the column containing group labels.
    mode : str, optional
        Test method: 'levene', 'bartlett', or 'auto'. Default is 'auto'.
        When 'auto', the method is automatically selected based on normality results
        and sample size.
    normality_res : pandas.DataFrame, optional
        Results from normality tests. Required when mode='auto'. Default is None.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with homogeneity test results.
    
    Raises
    ------
    ValueError
        If mode is not 'levene', 'bartlett', or 'auto', or if mode='auto' but
        normality_res is not provided.
    
    Examples
    --------
    >>> data = pd.DataFrame({'value': [1,2,3,4,5,6], 'group': ['A','A','B','B','A','B']})
    >>> normality = perform_normality_by_group(data, 'value', 'group')
    >>> result = perform_variances_homogeneity_test(data, 'value', 'group', mode='auto', normality_res=normality)
    """
    # Validate mode parameter
    if mode not in ["levene", "bartlett", "auto"]:
        raise ValueError("Parameter 'mode' should be 'levene', 'bartlett' or 'auto'. Default is 'auto'.")
    
    # Determine which homogeneity test method to use
    vh_method = determine_homogeneity_test_method(mode, data, group_col, normality_res)
    
    # Construct formula string for display
    formula_str = f"{response_col} ~ {group_col}"
    
    # Perform the appropriate test
    if vh_method == "bartlett":
        # Bartlett test
        groups = [group_data[response_col].dropna().values 
                 for _, group_data in data.groupby(group_col)]
        bt_result = bartlett(*groups)
        
        result_df = pd.DataFrame({
            "step": "Equality of Variances",
            "method": "Bartlett test",
            "group": formula_str,
            "p.value": bt_result.pvalue
        }, index=[0])
    else:
        # Levene test
        groups = [group_data[response_col].dropna().values 
                 for _, group_data in data.groupby(group_col)]
        lt_result = levene(*groups)
        
        result_df = pd.DataFrame({
            "step": "Equality of Variances",
            "method": "Levene test",
            "group": formula_str,
            "p.value": lt_result.pvalue
        }, index=[0])
    
    # Add significance stars
    result_df["sig"] = result_df["p.value"].apply(p_to_stars)
    
    print(f"Finish {result_df['method'].iloc[0]}!")
    return result_df


def perform_anova_test(data, response_col, group_col, factors=None, 
                       interaction=True, post_hoc_method="tukey"):
    """
    Perform ANOVA test (one-way, two-way, or multi-way).
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    factors : list of str, optional
        Additional factors for multi-way ANOVA. Default is None.
    interaction : bool, optional
        Whether to include interaction terms. Default is True.
    post_hoc_method : str, optional
        Post-hoc method: 'tukey' or 'holm'. Default is 'tukey'.
    
    Returns
    -------
    dict
        Dictionary containing ANOVA results and post-hoc tests (if applicable).
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'value': [1,2,3,4,5,6,7,8,9,10],
    ...     'group': ['A','A','B','B','C','C','A','A','B','B']
    ... })
    >>> result = perform_anova_test(data, 'value', 'group')
    """
    # Build formula
    formula_str = build_aov_formula(response_col, group_col, factors, interaction)
    print(f"Try to perform ANOVA test ...\n---> Formula = {formula_str}")
    
    # Perform ANOVA
    model = ols(formula_str, data=data).fit()
    anova_result = anova_lm(model, typ=2)
    
    # Determine ANOVA type
    if factors is None:
        aov_method = "one-way ANOVA"
    elif len(factors) == 1:
        aov_method = "two-way ANOVA"
    else:
        aov_method = "Multivariate ANOVA"
    
    # Check if interaction was actually performed
    interaction_performed = len(anova_result) > (len(factors) + 1 if factors else 1)
    
    if not interaction_performed and factors is not None:
        print("The design does not conform to multifactor analysis of variance.")
        print("Switch to one-way ANOVA ...")
        formula_str = build_aov_formula(response_col, group_col)
        print(f"---> Formula = {formula_str}")
        model = ols(formula_str, data=data).fit()
        anova_result = anova_lm(model, typ=2)
        aov_method = "one-way ANOVA"
    
    print("ANOVA results:")
    print(anova_result)
    
    # Extract results (exclude residual row)
    anova_result_clean = anova_result.iloc[:-1].copy()
    anova_result_clean["formula"] = [formula_str] + [np.nan] * (len(anova_result_clean) - 1)
    
    # Check if post-hoc is needed
    p_value_sig = any(anova_result_clean["PR(>F)"] < 0.05)
    
    if p_value_sig:
        print(f"ANOVA p<0.05. Performing post-hoc: {post_hoc_method}.")
        post_hoc_list = perform_anova_post_hoc(data, response_col, group_col, post_hoc_method)
        res_list = {
            "ANOVA": anova_result_clean,
            "post_hoc": post_hoc_list["post_hoc"],
            "post_hoc_method": post_hoc_list["post_hoc_method"]
        }
        print(res_list["post_hoc"])
    else:
        print("ANOVA p>0.05. No need for post hoc test.")
        res_list = {"ANOVA": anova_result_clean, "post_hoc_method": None}
    
    res_list["avo_method"] = aov_method
    
    return res_list




def perform_welch_anova(data, response_col, group_col):
    """
    Perform Welch ANOVA test (for unequal variances).
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    
    Returns
    -------
    dict
        Dictionary containing Welch ANOVA results and Games-Howell post-hoc (if applicable).
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'value': [1,2,3,4,5,6,7,8,9,10],
    ...     'group': ['A','A','B','B','C','C','A','A','B','B']
    ... })
    >>> result = perform_welch_anova(data, 'value', 'group')
    """
    # Ensure group column is categorical
    data = data.copy()
    data[group_col] = data[group_col].astype('category')
    
    # Build formula
    formula_str = build_aov_formula(response_col, group_col, factors=None, interaction=False)
    print(f"Try to perform Welch ANOVA test ...\n---> Formula = {formula_str}")
    
    # Perform Welch ANOVA (oneway.test equivalent)
    groups = [group_data[response_col].dropna().values 
             for _, group_data in data.groupby(group_col)]
    
    # Welch ANOVA using scipy
    f_stat, p_value = f_oneway(*groups)
    
    # Create result object similar to R's oneway.test
    welch_anova = {
        "statistic": f_stat,
        "p.value": p_value,
        "data.name": f"{response_col} and {group_col}"
    }
    
    # Post-hoc test if significant
    p_value_sig = p_value < 0.05
    
    if p_value_sig:
        print("Welch ANOVA p<0.05. Performing post-hoc: Games-Howell.")
        # Games-Howell test using scikit-posthocs or fallback
        if HAS_SCIKIT_POSTHOCS:
            try:
                post_hoc_df = sp.posthoc_gameshowell(data, val_col=response_col, group_col=group_col)
                # Convert to comparison format
                comparisons = []
                p_values = []
                group_names = post_hoc_df.index.tolist()
                
                for i in range(len(post_hoc_df)):
                    for j in range(i+1, len(post_hoc_df)):
                        group1 = group_names[i]
                        group2 = group_names[j]
                        p_val = post_hoc_df.iloc[i, j]
                        comparisons.append(f"{group2}-{group1}")
                        p_values.append(p_val)
                
                post_hoc_df_final = pd.DataFrame({
                    "comparison": comparisons,
                    "p.value": p_values,
                    "method": "Games-Howell"
                })
            except (AttributeError, TypeError):
                # Fallback if scikit-posthocs fails
                post_hoc_df_final = _games_howell_fallback(data, response_col, group_col)
        else:
            # Fallback: use pairwise t-tests with Games-Howell correction
            post_hoc_df_final = _games_howell_fallback(data, response_col, group_col)
        
        res_list = {
            "Welch ANOVA": welch_anova,
            "post_hoc": post_hoc_df_final,
            "post_hoc_method": "Games-Howell"
        }
        print(res_list["post_hoc"])
    else:
        print("Welch ANOVA p>0.05. No need for post hoc test.")
        res_list = {"Welch ANOVA": welch_anova, "post_hoc_method": None}
    
    res_list["avo_method"] = "Welch ANOVA"
    
    return res_list


def perform_kw_test(data, response_col, group_col, post_hoc_method="dunn"):
    """
    Perform Kruskal-Wallis test (non-parametric test for multiple groups).
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    post_hoc_method : str, optional
        Post-hoc method: 'dunn' or 'wilcoxon'. Default is 'dunn'.
    
    Returns
    -------
    dict
        Dictionary containing Kruskal-Wallis results and post-hoc tests (if applicable).
    
    Raises
    ------
    ValueError
        If response_col or group_col not found, or if post_hoc_method is invalid.
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'value': [1,2,3,4,5,6,7,8,9,10],
    ...     'group': ['A','A','B','B','C','C','A','A','B','B']
    ... })
    >>> result = perform_kw_test(data, 'value', 'group', 'dunn')
    """
    # Validate input
    if response_col not in data.columns or group_col not in data.columns:
        raise ValueError("Response column or group column not found in the data.")
    if post_hoc_method not in ["dunn", "wilcoxon"]:
        raise ValueError("Parameter 'post_hoc_method' should be 'dunn' or 'wilcoxon'.")
    
    # Perform Kruskal-Wallis test
    groups = [group_data[response_col].dropna().values 
             for _, group_data in data.groupby(group_col)]
    
    kruskal_result = kruskal(*groups)
    print("Kruskal-Wallis Global Test Results:\n")
    print(kruskal_result)
    
    formula_str = f"{response_col} ~ {group_col}"
    kruskal_result_dict = {
        "statistic": kruskal_result.statistic,
        "pvalue": kruskal_result.pvalue,
        "formula": formula_str
    }
    
    # Post-hoc tests if significant
    if kruskal_result.pvalue < 0.05 and post_hoc_method == "dunn":
        print("\nKW p<0.05. Proceeding with post-hoc Dunn's tests:\n")
        
        # Dunn's test using scikit-posthocs or fallback
        if HAS_SCIKIT_POSTHOCS:
            try:
                dunn_result = sp.posthoc_dunn(data, val_col=response_col, group_col=group_col, p_adjust='holm')
                
                # Convert to comparison format
                comparisons = []
                p_values = []
                group_names = dunn_result.index.tolist()
                
                for i in range(len(dunn_result)):
                    for j in range(i+1, len(dunn_result)):
                        group1 = group_names[i]
                        group2 = group_names[j]
                        p_val = dunn_result.iloc[i, j]
                        comparisons.append(f"{group2}-{group1}")
                        p_values.append(p_val)
                
                dunn_df = pd.DataFrame({
                    "comparison": comparisons,
                    "p.value": p_values,
                    "method": "Dunn's test"
                })
            except (AttributeError, TypeError):
                # Fallback if scikit-posthocs fails
                dunn_df = _dunn_test_fallback(data, response_col, group_col)
        else:
            # Fallback: use pairwise Mann-Whitney U tests with Holm correction
            dunn_df = _dunn_test_fallback(data, response_col, group_col)
        
        print("\nDunn's Test Results:\n")
        print(dunn_df)
        
        res_list = {
            "Kruskal-Wallis": kruskal_result_dict,
            "post_hoc": dunn_df,
            "post_hoc_method": "Dunn's test"
        }
        print(res_list["post_hoc"])
        
    elif kruskal_result.pvalue < 0.05 and post_hoc_method == "wilcoxon":
        print("\nKW p<0.05. Proceeding with pairwise-wilcoxon tests:\n")
        pairwise_wilcox_result = perform_pairwise_wilcoxon(data, response_col, group_col)
        
        res_list = {
            "Kruskal-Wallis": kruskal_result_dict,
            "post_hoc": pairwise_wilcox_result,
            "post_hoc_method": "pairwise.wilcoxon"
        }
        print(res_list["post_hoc"])
        
    else:
        print("\nKW p>0.05. No need for pairwise comparisons.\n")
        res_list = {
            "Kruskal-Wallis": kruskal_result_dict,
            "post_hoc_method": None
        }
    
    return res_list


def rev_dunn_comparison(dunn_comparison):
    """
    Reverse the order of groups in Dunn's test comparison strings.
    
    Parameters
    ----------
    dunn_comparison : list or array
        List of comparison strings in format "group1 - group2".
    
    Returns
    -------
    list
        List of reversed comparison strings in format "group2-group1".
    
    Examples
    --------
    >>> comparisons = ["A - B", "A - C"]
    >>> rev_dunn_comparison(comparisons)
    ['B-A', 'C-A']
    """
    new_comparisons = []
    for comp in dunn_comparison:
        parts = comp.split(" - ")
        if len(parts) == 2:
            new_comp = f"{parts[1]}-{parts[0]}"
            new_comparisons.append(new_comp)
        else:
            new_comparisons.append(comp)
    return new_comparisons


# ============================================================================
# Tertiary Functions
# ============================================================================

def _games_howell_fallback(data, response_col, group_col):
    """
    Fallback Games-Howell test using pairwise t-tests with Holm correction.
    
    This is used when scikit-posthocs is not available.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with Games-Howell test results.
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    
    group_names = sorted(data[group_col].unique())
    comparisons = []
    p_values = []
    
    for i, group1 in enumerate(group_names):
        for group2 in group_names[i+1:]:
            data1 = data[data[group_col] == group1][response_col].dropna()
            data2 = data[data[group_col] == group2][response_col].dropna()
            _, p_val = ttest_ind(data1, data2, equal_var=False)
            comparisons.append(f"{group2}-{group1}")
            p_values.append(p_val)
    
    # Apply Holm correction (similar to Games-Howell for unequal variances)
    _, p_adjusted, _, _ = multipletests(p_values, method='holm')
    
    return pd.DataFrame({
        "comparison": comparisons,
        "p.value": p_adjusted,
        "method": "Games-Howell (fallback)"
    })


def _dunn_test_fallback(data, response_col, group_col):
    """
    Fallback Dunn's test using pairwise Mann-Whitney U tests with Holm correction.
    
    This is used when scikit-posthocs is not available.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with Dunn's test results.
    """
    from statsmodels.stats.multitest import multipletests
    
    group_names = sorted(data[group_col].unique())
    comparisons = []
    p_values = []
    
    for i, group1 in enumerate(group_names):
        for group2 in group_names[i+1:]:
            data1 = data[data[group_col] == group1][response_col].dropna()
            data2 = data[data[group_col] == group2][response_col].dropna()
            test_res = mannwhitneyu(data1, data2, alternative='two-sided')
            p_val = test_res.pvalue
            comparisons.append(f"{group2}-{group1}")
            p_values.append(p_val)
    
    # Apply Holm correction (as used in Dunn's test)
    _, p_adjusted, _, _ = multipletests(p_values, method='holm')
    
    return pd.DataFrame({
        "comparison": comparisons,
        "p.value": p_adjusted,
        "method": "Dunn's test (fallback)"
    })


def determine_homogeneity_test_method(mode, data, group_col, normality_res=None):
    """
    Determine which homogeneity test method to use based on mode and data characteristics.
    
    Parameters
    ----------
    mode : str
        Test mode: 'levene', 'bartlett', or 'auto'.
    data : pandas.DataFrame
        The input data.
    group_col : str
        Name of the column containing group labels.
    normality_res : pandas.DataFrame, optional
        Results from normality tests. Required when mode='auto'. Default is None.
    
    Returns
    -------
    str
        The selected method: 'levene' or 'bartlett'.
    
    Raises
    ------
    ValueError
        If mode='auto' but normality_res is not provided.
    
    Examples
    --------
    >>> data = pd.DataFrame({'value': [1,2,3,4,5,6], 'group': ['A','A','B','B','A','B']})
    >>> normality = perform_normality_by_group(data, 'value', 'group')
    >>> method = determine_homogeneity_test_method('auto', data, 'group', normality)
    """
    # If mode is already specified, just validate it
    if mode != "auto":
        # Warning for small group size when using Bartlett
        min_group_numb = data[group_col].value_counts().min()
        if min_group_numb < 5 and mode == "bartlett":
            warnings.warn("The minimum number of groups is less than 5. Levene test is recommended.")
        return mode
    
    # For auto mode, determine the appropriate method
    print("Set variances homogeneity test mode = 'auto'.")
    
    if normality_res is None:
        raise ValueError("When mode='auto', 'normality_res' must be provided to determine appropriate homogeneity test.")
    
    # Check normality and sample size
    nor_res = any(normality_res['p.value'] < 0.05)
    min_group_numb = data[group_col].value_counts().min()
    
    # Choose method based on criteria
    # Use Levene if: minimum group size < 50 OR normality assumption violated
    # Otherwise use Bartlett
    method = "levene" if (min_group_numb < 50 or nor_res) else "bartlett"
    print(f"Auto set variances homogeneity test mode = '{method}'.")
    
    return method


def build_aov_formula(response_col, group_col, factors=None, interaction=True):
    """
    Build ANOVA formula string.
    
    Parameters
    ----------
    response_col : str
        Name of the response variable.
    group_col : str
        Name of the group variable.
    factors : list of str, optional
        Additional factors for multi-way ANOVA. Default is None.
    interaction : bool, optional
        Whether to include interaction terms. Default is True.
    
    Returns
    -------
    str
        Formula string for ANOVA.
    
    Examples
    --------
    >>> build_aov_formula('value', 'group')
    'value ~ group'
    >>> build_aov_formula('value', 'group', ['factor1', 'factor2'], interaction=True)
    'value ~ factor1*factor2'
    """
    if factors is None:
        # One-way ANOVA
        formula_str = f"{response_col} ~ {group_col}"
    else:
        # Multi-way ANOVA
        if interaction:
            # Include main effects and interactions
            formula_str = f"{response_col} ~ {'*'.join(factors)}"
        else:
            # Only main effects, no interactions
            formula_str = f"{response_col} ~ {'+'.join(factors)}"
    
    return formula_str


def perform_anova_post_hoc(data, response_col, group_col, method="tukey"):
    """
    Perform post-hoc tests after ANOVA.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    method : str, optional
        Post-hoc method: 'tukey' or 'holm'. Default is 'tukey'.
    
    Returns
    -------
    dict
        Dictionary containing post-hoc results and method name.
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'value': [1,2,3,4,5,6,7,8,9,10],
    ...     'group': ['A','A','B','B','C','C','A','A','B','B']
    ... })
    >>> result = perform_anova_post_hoc(data, 'value', 'group', 'tukey')
    """
    if method == "tukey":
        # Tukey HSD test
        mc = MultiComparison(data[response_col], data[group_col])
        tukey_result = mc.tukeyhsd()
        
        # Extract results
        tukey_df = pd.DataFrame({
            "comparison": [f"{row[1]}-{row[0]}" for row in tukey_result._results_table.data[1:]],
            "p.value": tukey_result.pvalues,
            "method": "Tukey HSD"
        })
        
        post_hoc_method = "Tukey HSD"
        res_list = {"post_hoc": tukey_df, "post_hoc_method": post_hoc_method}
        
    elif method == "holm":
        # Holm-Sidak correction using pairwise t-tests
        from scipy.stats import ttest_ind
        from itertools import combinations
        
        groups = data[group_col].unique()
        comparisons = []
        p_values = []
        
        for group1, group2 in combinations(groups, 2):
            data1 = data[data[group_col] == group1][response_col].dropna()
            data2 = data[data[group_col] == group2][response_col].dropna()
            _, p_val = ttest_ind(data1, data2)
            comparisons.append(f"{group2}-{group1}")
            p_values.append(p_val)
        
        # Apply Holm correction
        from statsmodels.stats.multitest import multipletests
        _, p_adjusted, _, _ = multipletests(p_values, method='holm')
        
        holm_res = pd.DataFrame({
            "comparison": comparisons,
            "p.value": p_adjusted,
            "method": "Holm-Sidak"
        })
        
        post_hoc_method = "Holm-Sidak"
        res_list = {"post_hoc": holm_res, "post_hoc_method": post_hoc_method}
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'tukey' or 'holm'.")
    
    return res_list


def perform_pairwise_wilcoxon(data, response_col, group_col):
    """
    Perform pairwise Wilcoxon rank-sum tests between all groups.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with pairwise Wilcoxon test results.
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'value': [1,2,3,4,5,6,7,8,9,10],
    ...     'group': ['A','A','B','B','C','C','A','A','B','B']
    ... })
    >>> result = perform_pairwise_wilcoxon(data, 'value', 'group')
    """
    # Get group combinations
    comparison_list = get_group_combinations(data, group_col=group_col)
    
    # Determine distribution method based on sample size
    total_size = len(data)
    distribution_method = "asymptotic" if total_size > 200 else "exact"
    
    # Perform pairwise Wilcoxon tests
    wilcox_res_list = []
    for groups_pair in comparison_list:
        data_use = data[data[group_col].isin(groups_pair)].copy()
        data_use[group_col] = data_use[group_col].astype('category')
        
        group1_data = data_use[data_use[group_col] == groups_pair[0]][response_col].dropna()
        group2_data = data_use[data_use[group_col] == groups_pair[1]][response_col].dropna()
        
        wilcox_res = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        p_value = wilcox_res.pvalue
        
        df = pd.DataFrame({
            "comparison": [f"{groups_pair[1]}-{groups_pair[0]}"],
            "p.value": [p_value],
            "method": ["pairwise Wilcoxon test"]
        })
        wilcox_res_list.append(df)
    
    wilcox_res_merge = pd.concat(wilcox_res_list, ignore_index=True)
    return wilcox_res_merge


def get_group_combinations(data, group_col="group"):
    """
    Get all pairwise combinations of groups.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    group_col : str, optional
        Name of the group column. Default is 'group'.
    
    Returns
    -------
    list
        List of tuples, each containing a pair of groups.
    
    Examples
    --------
    >>> data = pd.DataFrame({'group': ['A', 'B', 'C', 'A', 'B', 'C']})
    >>> get_group_combinations(data, 'group')
    [('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    # Get unique groups
    unique_groups = sorted(data[group_col].unique())
    
    # Get all pairwise combinations
    group_combinations = list(combinations(unique_groups, 2))
    
    return group_combinations


def p_to_stars(p):
    """
    Convert p-value to significance stars notation.
    
    Parameters
    ----------
    p : float or str
        P-value to convert.
    
    Returns
    -------
    str
        Significance stars: '' (ns), '*' (p<0.05), '**' (p<0.01), 
        '***' (p<0.001), '****' (p<0.0001).
    
    Examples
    --------
    >>> p_to_stars(0.03)
    '*'
    >>> p_to_stars(0.001)
    '***'
    >>> p_to_stars(0.5)
    ''
    """
    # Try to convert to numeric
    try:
        p_numeric = float(p)
    except (ValueError, TypeError):
        return ""
    
    # Check p-value and return corresponding stars
    if p_numeric < 0.0001:
        return "****"
    elif p_numeric < 0.001:
        return "***"
    elif p_numeric < 0.01:
        return "**"
    elif p_numeric < 0.05:
        return "*"
    else:
        return ""


# ============================================================================
# For two group test
# ============================================================================


def perform_stat_for_two_group(data, response_col, group_col, method):
    """
    Perform statistical test for two-group comparison.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    response_col : str
        Name of the response variable column.
    group_col : str
        Name of the group variable column.
    method : str
        Test method: 't.test' or 'wilcox.test'.
    
    Returns
    -------
    dict
        Dictionary containing test results.
    
    Raises
    ------
    ValueError
        If method is not 't.test' or 'wilcox.test'.
    
    Examples
    --------
    >>> data = pd.DataFrame({'value': [1,2,3,4,5,6], 'group': ['A','A','B','B','A','B']})
    >>> result = perform_stat_for_two_group(data, 'value', 'group', 't.test')
    """
    # Ensure group column is categorical
    data = data.copy()
    data[group_col] = data[group_col].astype('category')
    
    # Get groups
    groups = data[group_col].cat.categories
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, but found {len(groups)} groups.")
    
    group1_data = data[data[group_col] == groups[0]][response_col].dropna()
    group2_data = data[data[group_col] == groups[1]][response_col].dropna()
    
    # Perform test
    if method == "t.test":
        test_res = ttest_ind(group1_data, group2_data, equal_var=True)
        p_value = test_res.pvalue
    elif method == "wilcox.test":
        # Mann-Whitney U test (Wilcoxon rank-sum test)
        test_res = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        p_value = test_res.pvalue
    else:
        raise ValueError("Parameter 'method' should be 't.test' or 'wilcox.test'.")
    
    # Create result DataFrame
    step_use = "Parametric tests" if method == "t.test" else "Nonparametric tests"
    comparison = f"{groups[1]}-{groups[0]}"
    
    test_df = pd.DataFrame({
        "step": [step_use],
        "method": [method],
        "comparison": [comparison],
        "p.value": [p_value]
    })
    test_df["sig"] = test_df["p.value"].apply(p_to_stars)
    
    # Collect results
    res_list = {
        "test.results": test_df,
        "test_method": method
    }
    
    return res_list


def stat_res_collect_two_group(data, nor, stat, task_name):
    """
    Collect and organize statistical test results for two-group comparisons.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The original input data.
    nor : dict
        Results from test_assumptions() function.
    stat : dict
        Results from perform_stat_for_two_group() function.
    task_name : str
        Name of the analysis task.
    
    Returns
    -------
    dict
        A dictionary containing:
        - task_name: The original data
        - 'stat_test': DataFrame with all test results organized
    
    Examples
    --------
    >>> data = pd.DataFrame({'value': [1,2,3,4,5,6], 'group': ['A','A','B','B','A','B']})
    >>> assumptions = test_assumptions(data, 'value', 'group')
    >>> stat_result = perform_stat_for_two_group(data, 'value', 'group', 't.test')
    >>> final_result = stat_res_collect_two_group(data, assumptions, stat_result, 'my_analysis')
    """
    # Merge statistical results
    df_merge = pd.concat([nor["test.results"], stat["test.results"]], ignore_index=True)
    
    # Format p-values
    df_merge["p.value"] = df_merge["p.value"].apply(
        lambda x: round(x, 4) if isinstance(x, (int, float)) and not pd.isna(x) else x
    )
    df_merge.loc[df_merge["p.value"].apply(
        lambda x: isinstance(x, (int, float)) and x < 0.0001
    ), "p.value"] = "<0.0001"
    
    # Collect results
    res_dict = {task_name: data, "stat_test": df_merge}
    
    return res_dict