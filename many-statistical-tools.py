import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def calculate_mean(data):
    """
    Calculates the mean of a dataset.

    Parameters:
    data (list or array): the dataset to calculate the mean of.

    Returns:
    float: the mean of the dataset.
    """
    return np.mean(data)

def calculate_median(data):
    """
    Calculates the median of a dataset.

    Parameters:
    data (list or array): the dataset to calculate the median of.

    Returns:
    float: the median of the dataset.
    """
    return np.median(data)

def calculate_mode(data):
    """
    Calculates the mode of a dataset.

    Parameters:
    data (list or array): the dataset to calculate the mode of.

    Returns:
    list: a list of modes of the dataset.
    """
    return stats.mode(data)

def calculate_variance(data):
    """
    Calculates the variance of a dataset.

    Parameters:
    data (list or array): the dataset to calculate the variance of.

    Returns:
    float: the variance of the dataset.
    """
    return np.var(data)

def calculate_standard_deviation(data):
    """
    Calculates the standard deviation of a dataset.

    Parameters:
    data (list or array): the dataset to calculate the standard deviation of.

    Returns:
    float: the standard deviation of the dataset.
    """
    return np.std(data)

def perform_t_test(data1, data2):
    """
    Performs a t-test to determine if there is a significant difference between the means of two datasets.

    Parameters:
    data1 (list or array): the first dataset to compare.
    data2 (list or array): the second dataset to compare.

    Returns:
    float: the t-statistic of the test.
    float: the p-value of the test.
    """
    return stats.ttest_ind(data1, data2)

def perform_z_test(data1, data2):
    """
    Performs a z-test to determine if there is a significant difference between the means of two datasets.

    Parameters:
    data1 (list or array): the first dataset to compare.
    data2 (list or array): the second dataset to compare.

    Returns:
    float: the z-statistic of the test.
    float: the p-value of the test.
    """
    return stats.ttest_ind(data1, data2)

def perform_chi_squared_test(data1, data2):
    """
    Performs a chi-squared test to determine if there is a significant association between two categorical variables.

    Parameters:
    data1 (list or array): the first dataset to compare.
    data2 (list or array): the second dataset to compare.

    Returns:
    float: the chi-squared statistic of the test.
    float: the p-value of the test.
    """
    return stats.chi2_contingency([data1, data2])

def perform_anova(*args):
    """
    Performs an analysis of variance to determine if there is a significant difference between the means of more than two datasets.

    Parameters:
    *args (list or array): the datasets to compare.

    Returns:
    float: the f-statistic of the test.
    float: the p-value of the test.
    """
    return stats.f_oneway(*args)

def calculate_correlation(data1, data2):
    """
    Calculates the correlation between two variables.

    Parameters:
    data1 (list or array): the first variable to correlate.
    data2 (list or array): the second variable to correlate.

    Returns:
    float: the correlation coefficient.
    """
    return np.corrcoef(data1, data2)[0][1]

def perform_linear_regression(data1, data2):
    """
    Performs a linear regression analysis to model the relationship between two variables.

    Parameters:
    data1 (list or array): the independent variable.
    data2 (list or array): the dependent variable.

    Returns:
    float: the slope of the regression line.
    float: the intercept of the regression line.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
    return slope, intercept

def perform_wilcoxon_rank_sum_test(data1, data2):
    """
    Performs a Wilcoxon rank-sum test to determine if there is a significant difference between the medians of two datasets.

    Parameters:
    data1 (list or array): the first dataset to compare.
    data2 (list or array): the second dataset to compare.

    Returns:
    float: the test statistic of the test.
    float: the p-value of the test.
    """
    return stats.ranksums(data1, data2)

def perform_kruskal_wallis_test(*args):
    """
    Performs a Kruskal-Wallis test to determine if there is a significant difference between the medians of more than two datasets.

    Parameters:
    *args (list or array): the datasets to compare.

    Returns:
    float: the H-statistic of the test.
    float: the p-value of the test.
    """
    return stats.kruskal(*args)

def perform_friedman_test(*args):
    """
    Performs a Friedman test to determine if there is a significant difference between the medians of more than two datasets, when the data is repeated measures.

    Parameters:
    *args (list or array): the datasets to compare.

    Returns:
    float: the test statistic of the test.
    float: the p-value of the test.
    """
    return stats.friedmanchisquare(*args)

def perform_mcnemar_test(data):
    """
    Performs a McNemar test to determine if there is a significant difference between two proportions.

    Parameters:
    data (array): a contingency table of the data, where the rows are the two tests being compared and the columns are the number of observations that fall into each of the four categories.

    Returns:
    float: the test statistic of the test.
    float: the p-value of the test.
    """
    return stats.mcnemar(data)

def perform_students_t_test_for_paired_samples(data1, data2):
    """
    Performs a Student's t-test for paired samples to determine if there is a significant difference between the means of two related datasets.

    Parameters:
    data1 (list or array): the first dataset to compare.
    data2 (list or array): the second dataset to compare.

    Returns:
    float: the t-statistic of the test.
    float: the p-value of the test.
    """
    return stats.ttest_rel(data1, data2)

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculates the confidence interval of a dataset.

    Parameters:
    data (list or array): the dataset to calculate the confidence interval of.
    confidence (float): the level of confidence for the interval (default is 0.95).

    Returns:
    tuple: a tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(data)
    m = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m - h, m + h

def calculate_p_value(data, null_mean=0):
    """
    Calculates the p-value of a dataset assuming a null hypothesis.

    Parameters:
    data (list or array): the dataset to calculate the p-value of.
    null_mean (float): the null hypothesis mean (default is 0).

    Returns:
    float: the p-value of the test.
    """
    t_stat, p_val = stats.ttest_1samp(data, null_mean)
    return p_val

def calculate_power_of_test(data, effect_size, alpha=0.05):
    """
    Calculates the power of a statistical test assuming a certain effect size and level of significance.

    Parameters:
    data (list or array): the dataset to perform the test on.
    effect_size (float): the effect size to detect.
    alpha (float): the level of significance (default is 0.05).

    Returns:
    float: the power of the test.
    """
    n = len(data)
    std = np.std(data)
    se = std / np.sqrt(n)
    z = (effect_size * np.sqrt(n)) / se
    power = 1 - stats.norm.cdf(z - stats.norm.ppf(1 - alpha / 2))
    return power

def visualize_data(data, plot_type='histogram', **kwargs):
    """
    Visualizes the data using a given type of plot.

    Parameters:
    data (list or array): the dataset to visualize.
    plot_type (str): the type of plot to use (default is 'histogram').
    **kwargs: additional arguments to pass to the plotting function.

    Returns:
    None.
    """
    if plot_type == 'histogram':
        plt.hist(data, **kwargs)
    elif plot_type == 'boxplot':
        plt.boxplot(data, **kwargs)
    elif plot_type == 'scatter':
        plt.scatter(*data, **kwargs)
    elif plot_type == 'line':
        plt.plot(*data, **kwargs)
    else:
        raise ValueError(f'Invalid plot type: {plot_type}')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform statistical analysis on a dataset.')
    parser.add_argument('input_file', type=str, help='the path to the input file')
    parser.add_argument('-o', '--output_file', type=str, help='the path to the output file')
    parser.add_argument('-t', '--test', type=str, help='the type of statistical test to perform')
    parser.add_argument('-c', '--confidence', type=float, help='the level of confidence for the confidence interval')
    parser.add_argument('-s', '--size', type=float, help='the effect size for the power of test')
    parser.add_argument('-a', '--alpha', type=float, help='the level of significance for the power of test')
    parser.add_argument('-p', '--plot', type=str, help='the type of plot to visualize the data')
    args = parser.parse_args()

    data = pd.read_csv(args.input_file, header=None).values.flatten()

    if args.test:
        if args.test == 'mean':
            result = calculate_mean(data)
            print(f'Mean: {result:.2f}')
        elif args.test == 'median':
            result = calculate_median(data)
            print(f'Median: {result:.2f}')
        elif args.test == 'mode':
            result = calculate_mode(data)
            print(f'Mode: {result.mode[0]}')
        elif args.test == 'variance':
            result = calculate_variance(data)
            print(f'Variance: {result:.2f}')
        elif args.test == 'standard_deviation':
            result = calculate_standard_deviation(data)
            print(f'Standard deviation: {result:.2f}')
        elif args.test == 't_test':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            t_stat, p_val = perform_t_test(data1, data2)
            print(f'T-test: t-statistic = {t_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'z_test':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            z_stat, p_val = perform_z_test(data1, data2)
            print(f'Z-test: z-statistic = {z_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'chi_squared_test':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            chi2_stat, p_val, _, _ = perform_chi_squared_test(data1, data2)
            print(f'Chi-squared test: chi-squared statistic = {chi2_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'anova':
            num_groups = 3
            group_size = len(data) // num_groups
            groups = [data[i*group_size:(i+1)*group_size] for i in range(num_groups)]
            f_stat, p_val = perform_anova(*groups)
            print(f'ANOVA: f-statistic = {f_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'correlation':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            corr = calculate_correlation(data1, data2)
            print(f'Correlation: {corr:.2f}')
        elif args.test == 'linear_regression':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            slope, intercept = perform_linear_regression(data1, data2)
            print(f'Linear regression: y = {slope:.2f}x + {intercept:.2f}')
        elif args.test == 'wilcoxon_rank_sum_test':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            test_stat, p_val = perform_wilcoxon_rank_sum_test(data1, data2)
            print(f'Wilcoxon rank-sum test: test statistic = {test_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'kruskal_wallis_test':
            num_groups = 3
            group_size = len(data) // num_groups
            groups = [data[i*group_size:(i+1)*group_size] for i in range(num_groups)]
            h_stat, p_val = perform_kruskal_wallis_test(*groups)
            print(f'Kruskal-Wallis test: H-statistic = {h_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'friedman_test':
            num_groups = 3
            num_repeats = len(data) // (num_groups**2)
            data = data.reshape(num_repeats, num_groups, num_groups)
            test_stat, p_val = perform_friedman_test(*[data[i,:,:] for i in range(num_repeats)])
            print(f'Friedman test: test statistic = {test_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'mcnemar_test':
            data = data.reshape(-1, 2)
            test_stat, p_val = perform_mcnemar_test(data)
            print(f'McNemar test: test statistic = {test_stat:.2f}, p-value = {p_val:.4f}')
        elif args.test == 'students_t_test_for_paired_samples':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            t_stat, p_val = perform_students_t_test_for_paired_samples(data1, data2)
            print(f"Student's t-test for paired samples: t-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")
        else:
            print(f'Invalid test: {args.test}')
            sys.exit(1)

    if args.confidence:
        ci = calculate_confidence_interval(data, args.confidence)
        print(f'{args.confidence*100:.0f}% confidence interval: ({ci[0]:.2f}, {ci[1]:.2f})')

    if args.size and args.alpha:
        power = calculate_power_of_test(data, args.size, args.alpha)
        print(f'Power of test: {power:.2f}')

    if args.plot:
        if args.plot == 'histogram':
            visualize_data(data, plot_type='histogram')
        elif args.plot == 'boxplot':
            visualize_data(data, plot_type='boxplot')
        elif args.plot == 'scatter':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            visualize_data((data1, data2), plot_type='scatter')
        elif args.plot == 'line':
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            visualize_data((data1, data2), plot_type='line')
        else:
            print(f'Invalid plot type: {args.plot}')
            sys.exit(1)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(','.join(str(x) for x in data))

           

           


