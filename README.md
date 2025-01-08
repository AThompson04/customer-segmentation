# Customer Segmentation for Marketing Analysis

## Data
The dataset for this project can be found on [Kaggle](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis) under the CC0: Public Domain license.

The dataset consists of simulated customer data, containing demographic and behavioural information for 1 000 customer records. 
The data is unlabelled, therefore unsupervised machine learning methods will be used to discover patterns and relationships in the data.

## Objectives
The main objective of this project is:
> To identify different customer segments based on the customer's demographic and behavioural information. The customer segments can the be used to develop targeted marketing strategies for each segment

To achieve this objective, it was further broken down into the following four technical sub-objectives:
1. To perform an exploratory data analysis.
2. Data cleaning.
3. To use an unsupervised machine learning method, K-means, to group that customers into different segments based on their demographic and behavioural information.
4. Provide summaries for each customer segment to assist in developing targeted marketing strategies.

## Main Insights
From the exploratory data analysis, we found that:
- The continuous variables, i.e. *age*, *income*, *spending_score*, *last_purchase_amount* and *purchase_frequency*, appear to be somewhat evenly distributed.
<figure>
    <img src="/assets/cont_var_dist.png">
    <figcaption>Figure 1: Distribution of Continuous Variables</figcaption>
</figure>
- There are no strong relationships between any of the continuous variables mentioned previously. All correlations range between -0.06 and 0.06.
- There are no strong relationships between the continuous variables mentioned above and the categorical variables, *gender* and *preferred_category*. However, there seems to be a slight tendency for males to purchase more frequently and that females last purchase amount tends to be lower than males or others. 

## Data Cleaning and Engineered Features
The following steps were taken to prepare the data for model selection:
- The dataset was checked for any missed values and duplicated customer data. No missing values or duplicate values were found.
- The continuous variables were normalised using Sklearn's StandardScaler function. The Standard Scaler function uses z-score normalisation by transforming the data has a mean of zero and standard deviation of one.
- The StandardScaler instance was saved to be used to convert the normalised data back to its original form after cluster has occurred to provide useful insight about each cluster.
- The categorical variables *gender* and *preferred_category* were transformed into categorical data using One-Hot Encoding.

## Model Selection and Methodology
### Model Selection:
Due to the unlabelled nature of the dataset an unsupervised machine learning method was selected to group the data points based on patterns and relationships found in the data. K-means clustering was selected as it is used to group data into groups based on the similarity of the data points, ability to handle large datasets quickly and its computational efficiency.

### Methodology:
- The whole dataset was used to model the data, as there was no need to create a train-test split for K-means clustering.
- To determine the number of clusters the following was done:
  - All instances of Sklearn.cluster's KMeans used the random state 42.
  - The distortion score, silhouette score and the fit time were calculated for each instance of KMeans for the different values of k i.e. number of clusters.
  - The values of k tested were two to twelve.
  - An elbow plot with distortion score and fit time, and a line graph for silhouette scores for the values of k were created.
  - The value of k at the elbow of elbow plot with a higher silhouette score would be used for the analysis, only if the silhouette score is less than the silhouetted coefficient value of each cluster for the specific k value.
- Once the number of cluster has been chosen, the data will be sorted into clusters using Sklearn.cluster's KMeans function.
- A principal component analysis using two components will be preformed so that the clustering can be analysed visually using a scatterplot.
- All normalised data will be transformed to its original scale, so meaningful cluster analysis can be done.

## Results and Cluster Analysis
### Results:
 As seen in Figure 2, there is no clear elbow however at k=6 there is a significant drop in fit time which corresponds to one of the highest silhouette scores in Figure 3. There is no steady increase in the silhouette score after k=6, the silhouette score fluctuates.

<figure>
    <img src="/assets/elbow.png" style="width: 75%; height: auto;">
    <figcaption>Figure 2: Distortion Score Elbow for KMeans Clustering</figcaption>
</figure>

<figure>
    <img src="/assets/sil.png" style="width: 75%; height: auto;">
    <figcaption>Figure 3: Silhouette Score Analysis for Optimal K</figcaption>
</figure>
Therefore, using the elbow plot, Figure 2, and the silhouette score per cluster, Figure 3, a six cluster approach was selected.

<figure>
    <img src="/assets/sil_vis.png" style="width: 75%; height: auto;">
    <figcaption>Figure 4: Silhouette Plot of KMeans Clustering for 1000 Samples in 6 Centers</figcaption>
</figure>

When analysing the individual silhouette scores across each cluster in Figure 4, each cluster's silhouette score is larger than the average silhouette score. This suggests that the clusters are well separated and internally consistent.

However the overall silhouette score is low, indicating that the clusters overlap this is evident when viewing the clusters using PCA in Figure 5.

<figure>
    <img src="/assets/pca_clusters.png" style="width: 75%; height: auto;">
    <figcaption>Figure 5: Visualising the Clusters after Principal Component Analysis</figcaption>
</figure>

In an attempt to improve the quality of the clustering the categorical variables *gender* and *preferred_category* were removed, however there was no significant increase in the quality of the clustering. 

### Cluster Analysis:

