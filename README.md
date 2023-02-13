
# Many Statistical Tools

This is a Python script that performs statistical analysis on a dataset. It includes functions for calculating various statistics such as the mean, median, and mode, as well as performing various statistical tests such as t-tests, chi-squared tests, and ANOVA. It also includes functions for calculating confidence intervals and p-values, as well as visualizing data using histograms, box plots, scatter plots, and line graphs.

## Requirements

-   Python 3.x
-   pandas
-   numpy
-   scipy
-   matplotlib

## Usage

The script is run from the command line, and takes the following arguments:

    usage: dataset_analysis.py [-h] [-o OUTPUT_FILE] [-t TEST] [-c CONFIDENCE] [-s SIZE] [-a ALPHA] [-p PLOT] input_file
    
    Perform statistical analysis on a dataset.
    
    positional arguments:
      input_file            the path to the input file
    
    optional arguments:
      -h, --help  show this help message and exit
      -o OUTPUT_FILE, --output_file OUTPUT_FILE  the path to the output file
      -t TEST, --test TEST  the type of statistical test to perform
      -c CONFIDENCE, --confidence CONFIDENCE   the level of confidence for the confidence interval
      -s SIZE, --size SIZE  the effect size for the power of test
      -a ALPHA, --alpha ALPHA the level of significance for the power of test
      -p PLOT, --plot PLOT  the type of plot to visualize the data 

The `input_file` argument is required, and specifies the path to the input file containing the dataset to analyze. The file should be a CSV file with each row representing an observation, and each column representing a variable.

The `-o` or `--output_file` argument is optional, and specifies the path to the output file where the results of the analysis will be written. If this argument is not specified, the results will be printed to the console.

The `-t` or `--test` argument is optional, and specifies the type of statistical test to perform. The available options are:

-   `mean`: calculate the mean of the dataset
-   `median`: calculate the median of the dataset
-   `mode`: calculate the mode of the dataset
-   `variance`: calculate the variance of the dataset
-   `standard_deviation`: calculate the standard deviation of the dataset
-   `t_test`: perform a t-test to compare the means of two datasets
-   `z_test`: perform a z-test to compare the means of two datasets
-   `chi_squared_test`: perform a chi-squared test to compare the association between two categorical variables
-   `anova`: perform an analysis of variance to compare the means of more than two datasets
-   `correlation`: calculate the correlation coefficient between two variables
-   `linear_regression`: perform a linear regression analysis to model the relationship between two variables
-   `wilcoxon_rank_sum_test`: perform a Wilcoxon rank-sum test to compare the medians of two datasets
-   `kruskal_wallis_test`: perform a Kruskal-Wallis test to compare the medians of more than two datasets
-   `friedman_test`: perform a Friedman test to compare the medians of more than two datasets with repeated measures
-   `mcnemar_test`: perform a McNemar test to compare two proportions
-   `students_t_test_for_paired_samples`: perform a Student's t-test for paired samples to compare the means of two related datasets

The `-c` or `--confidence` argument is optional and specifies the level of confidence for the output. It is a floating-point number between 0 and 1, with a default value of 0.5. A higher confidence level will result in fewer false positives, but also a higher likelihood of false negatives.

If you want to save the output to a file, you can use the '-o' or '--output_file' option and specify the path to the output file.

You can also visualize the data using a given type of plot by using the '-p' or '--plot' option and specifying the type of plot. The supported plot types are 'histogram', 'boxplot', 'scatter', and 'line'. You can pass additional arguments to the plotting function using the '**kwargs' syntax.

Here are some examples of how to use the script:

1.  Calculate the mean of a dataset:
    
    python stats.py data.csv -t mean
    
2.  Calculate the confidence interval of a dataset with a 90% confidence level:

    python stats.py data.csv -c 0.9
    
3.  Visualize the data using a scatter plot:

    python stats.py data.csv -p scatter -c 'red'
    
4.  Perform a t-test on two datasets:
    
    python stats.py data.csv -t t_test
    
This will split the dataset in half and perform a t-test on the two resulting datasets.

For more information about the options and arguments of the script, you can use the '-h' or '--help' option.
