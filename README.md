# Statistics and Trends Assignment

Student Name: Karthik Guntumadugu

Student ID: 24086285

----


This is the completed template file for the statistics and trends assignment.
It analyses the Heart Disease Dataset to classify patients into 'No Disease' and 'Disease' categories.
The file includes a relational plot (line), categorical plot (bar), statistical plot (heatmap),
and statistical moment analysis for the 'age' column. 


Dataset: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### Discussion of Statistical Moments for `age`

#### 1. Mean = 54.43
The mean age of 54.43 years indicates the average age of patients in the dataset. This suggests that the population studied is predominantly middle-aged, which aligns with the context of heart disease, as risk typically increases with age. A mean of 54.43 implies that the dataset captures a group where cardiovascular issues are more prevalent, reflecting a realistic sample for studying heart disease prevalence. Since this is an average, it’s influenced by all values, including potential outliers, so it’s a central tendency measure providing a baseline for comparison with other features like cholesterol or blood pressure.

#### 2. Standard Deviation = 9.07
The standard deviation of 9.07 years measures the spread of ages around the mean. This moderate value indicates that most patients’ ages fall within approximately 9 years of 54.43 (i.e., roughly 45 to 63 years, assuming a normal distribution). A standard deviation of this magnitude suggests a relatively tight clustering of ages, with limited extreme values (e.g., very young or very old patients). In the context of heart disease, this implies the dataset focuses on a consistent age range where risk is significant, aiding in targeted analysis without excessive variability skewing results.

#### 3. Skewness = -0.25
Skewness of -0.25 reflects the asymmetry of the age distribution. A negative value indicates a left skew, where the left tail (younger ages) is slightly longer or fatter than the right tail (older ages). However, with a skewness of -0.25, the deviation from symmetry is mild (typically, |skew| < 0.5 is considered nearly symmetric). This suggests that while there are slightly more younger patients (e.g., below 54) than extremely old ones, the distribution is fairly balanced. In heart disease analysis, a slight left skew could imply early onset cases are present but not dominant, aligning with known risk patterns where incidence rises post-middle age.

#### 4. Excess Kurtosis = -0.53
Excess kurtosis of -0.53 describes the tailedness of the age distribution relative to a normal distribution (where excess kurtosis = 0). A negative value indicates a platykurtic distribution, meaning it has thinner tails and a flatter peak compared to a Gaussian curve. With a kurtosis of -0.53, the distribution is moderately platykurtic (|kurtosis| < 1 is subtle), suggesting fewer extreme ages (very young or old) and a broader central peak. For heart disease, this implies a consistent age range with fewer outliers, supporting a focus on middle-aged patients without significant anomalies that might complicate modeling.

---

### Interpretation in Context
For the `age` attribute, the mean of 54.43 and standard deviation of 9.07 suggest a middle-aged cohort with moderate variability, ideal for studying heart disease prevalence. The skewness of -0.25 indicates a slight left skew, hinting at a modest presence of younger patients, possibly reflecting early-onset cases, though the distribution remains nearly symmetric. The excess kurtosis of -0.53 points to a platykurtic shape, with a flatter peak and thinner tails, implying a stable age range without extreme outliers. Together, these moments suggest a well-distributed sample for classification, where age is a key but not overly varied factor. The data was **left skewed** (mildly, as -0.25 is close to 0) and **platykurtic**, indicating a broader, less peaked spread suitable for robust statistical analysis and machine learning without significant skew or tail distortions affecting model performance.

---
