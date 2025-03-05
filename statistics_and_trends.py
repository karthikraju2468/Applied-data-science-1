"""
This is the completed template file for the statistics and trends assignment.
It analyses the Heart Disease Dataset to classify patients into 'No Disease' and 'Disease' categories.
The file includes a relational plot (line), categorical plot (bar), statistical plot (heatmap),
and statistical moment analysis for the 'age' column. 


Student Name: Karthik Guntumadugu

Student ID: 24086285
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def plot_relational_plot(df):
    """Create a line plot showing mean cholesterol levels by age group and heart disease status.

    Args:
        df (pd.DataFrame): The preprocessed Heart Disease dataset.

    Returns:
        None: Saves the plot as 'relational_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    age_bins = pd.cut(df['age'], bins=range(20, 80, 10), right=False)
    df_grouped = df.groupby([age_bins, 'target'])['chol'].mean().unstack()
    df_grouped.plot(kind='line', ax=ax, color=['#800000', '#008080'], marker='o', linewidth=2)

    # Plot Settings
    ax.set_title('Mean Cholesterol by Age Group and Heart Disease Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Age Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Cholesterol (mg/dl)', fontsize=14, fontweight='bold')
    ax.legend(['No Disease', 'Disease'], title='Heart Disease', fontsize=12, title_fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add data values
    for i in range(len(df_grouped)):
        for j in range(2):
            if pd.notna(df_grouped.iloc[i, j]):
                ax.text(i, df_grouped.iloc[i, j], f'{df_grouped.iloc[i, j]:.0f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()


def plot_categorical_plot(df):
    """Create a bar plot showing the distribution of chest pain types by heart disease status.

    Args:
        df (pd.DataFrame): The preprocessed Heart Disease dataset.

    Returns:
        None: Saves the plot as 'categorical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='cp', hue='target', ax=ax, palette=['#800000', '#008080'], edgecolor='black')

    # Plot Settings
    ax.set_title('Distribution of Chest Pain Types by Heart Disease Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Chest Pain Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'], rotation=45, ha='right', fontsize=12)
    ax.legend(['No Disease', 'Disease'], title='Heart Disease', fontsize=12, title_fontsize=12)
    plt.yticks(fontsize=12)

    # Add data values
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close()


def plot_statistical_plot(df):
    """Create a correlation heatmap of numerical features in the dataset.

    Args:
        df (pd.DataFrame): The preprocessed Heart Disease dataset.

    Returns:
        None: Saves the plot as 'statistical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='PuBu', fmt='.2f', linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    # Plot Settings
    ax.set_title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    ax.collections[0].colorbar.ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close()


def statistical_analysis(df, col: str):
    """Calculate the four statistical moments for a specified column.

    Args:
        df (pd.DataFrame): The preprocessed dataset.
        col (str): The column name to analyze.

    Returns:
        tuple: Mean, standard deviation, skewness, excess kurtosis.
    """
    mean = np.mean(df[col])
    stddev = np.std(df[col])
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the dataset by handling missing values and checking basic statistics.

    Args:
        df (pd.DataFrame): The raw Heart Disease dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    # Check basic statistics
    print("Dataset Head:\n------------------------------------------------\n", df.head())
    print("\nDataset Description:\n--------------------------------------------\n", df.describe())
    print("\nCorrelation Matrix:\n-------------------------------------------\n", df.corr())

    # Handle missing values (none expected, but included for robustness)
    df = df.dropna()

    # Ensure correct data types (e.g., categorical as integers)
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    for col in categorical_cols:
        df[col] = df[col].astype(int)

    return df


def writing(moments, col):
    """Print and interpret the statistical moments for a given column.

    Args:
        moments (tuple): Mean, standard deviation, skewness, excess kurtosis.
        col (str): The column name analyzed.

    Returns:
        None: Prints the analysis.
    """
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    # Interpret skewness
    skew = moments[2]
    kurt = moments[3]
    skew_desc = 'right' if skew > 0.2 else 'left' if skew < -0.2 else 'not'
    kurt_desc = 'leptokurtic' if kurt > 0.2 else 'platykurtic' if kurt < -0.2 else 'mesokurtic'
    print(f'The data was {skew_desc} skewed and {kurt_desc}.')


def main():
    """Main function to execute the data analysis and visualization pipeline."""
    df = pd.read_csv('heart.csv')
    df = preprocessing(df)

    # Chosen column for statistical analysis
    col = 'age'  

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()