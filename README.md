# Customer Segmentation for Marketing Analysis

## Data
The dataset for this project can be found on [Kaggle](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis) under the CC0: Public Domain license.

The dataset consists of simulated customer data, containing demographic and behavioural information for 1 000 customer records. 
The data is unlabelled, therefore unsupervised machine learning methods will be used to discover patterns and relationships in the data.

The dataset is mixed - it is made up of numeric and nominal categorical variables.

## Objectives
The main objective of this project is:
> To identify different customer segments based on the customer's demographic and behavioural information. The customer segments can the be used to develop targeted marketing strategies for each segment

To achieve this objective, it was further broken down into the following four technical sub-objectives:
1. Perform an exploratory data analysis.
2. Perform data cleaning.
3. Determine which unsupervised machine learning method should be used to group the customers into different segments based on their demographic and behavioural information.
4. Provide summaries for each customer segment to assist in developing targeted marketing strategies.

## Main Insights
From the exploratory data analysis, we found that:
- The continuous variables, i.e. *age*, *income*, *spending_score*, *last_purchase_amount* and *purchase_frequency*, appear to be somewhat evenly distributed, see in Figure 1 below.
<figure align="center">
    <img src="/assets/cont_var_dist.png">
    <figcaption>Figure 1: Distribution of Continuous Variables</figcaption>
</figure>
</br>
</br>

- There are no strong relationships between any of the continuous variables mentioned previously. All correlations range between -0.06 and 0.06.
- There are no strong relationships between the continuous variables mentioned above and the categorical variables, *gender* and *preferred_category*. However, there seems to be a slight tendency for males to purchase more frequently and that females last purchase amount tends to be lower than males or others. 

## Data Cleaning and Engineered Features
The following steps were taken to prepare the data for model selection:
- The dataset was checked for any missed values and duplicated customer data. No missing values or duplicate values were found.
- The continuous variables were normalised using *Sklearn*'s *StandardScaler* function. The Standard Scaler function uses z-score normalisation by transforming the data has a mean of zero and standard deviation of one.
- The nominal categorical variables *gender* and *preferred_category* were transformed using One-Hot Encoding.

In addition to the cleaned dataset which was used for the K-Means approach, the original data was used for the following clustering methods:
1. LLM and K-Means
2. Hybrid approach to LLM and K-Means
3. K-Prototypes
Any normalisation for the for the previously mentioned approaches was done using *Sklearn*'s *StandardScaler* function.

## Model Selection and Methodology
Due to the unlabelled nature of the dataset an unsupervised machine learning method was selected to group the data points based on patterns and relationships found in the data. Different clustering methods were explored to determine which provided the most meaningful customer segmentation.

The following clustering methods were explored:
1. K-Means
2. LLM and K-Means
3. A Hybrid approach to LLM and K-Means
4. K-Prototypes 

### Methodology:
The following methodology was used for all approaches:
- The whole dataset was used to model the data, as there was no need to create a train-test split for K-means clustering.
- The optimum number of clusters was determined by:
  - Finding the elbow or determine the most optimal number of clusters using the distortion score scree plot.
  - Calculating the silhouette score for a range of different clusters.
  - If there was no clear decision as to the optimal number of clusters in the scree plot, there was no elbow, and there is quite small fluctuation with the clusters' silhouette scores. Then consider using silhouette plots for a range of different clusters. The aim would be to find the smallest cluster size where all individual cluster's silhouette coefficient values are larger than the average silhouette score to avoid poorly separated clusters and incorrectly classified data, and that there is no imbalance in cluster size.
  - Analysing the silhouette plot for the chosen number of clusters to ensure that each cluster silhouette score is larger than the average silhouette score, there are not many or any incorrectly classified data points and that the cluster sizes appear balanced.
- Cluster the data using the chosen clustering approach using random state 42.
- Preform a principal component analysis using three principal components. Plot 2D and 3D scatterplots coloured by cluster to visualise the clustering quality.
- Preform cluster analysis on the original data using the computed cluster labels. 

#### Additional Methodology for the Different Clustering Approaches:
##### 1. LLM and K-Means:
- The original data is converted into text after combining all variables for each customer, creating a 'customer profile'.
- The text is embedding using *sentence_transformers*'s *SentenceTransformer* function with the *paraphrase-MiniLM-L6-v2* model. The embeddings were normalised by the *SentenceTransformer* function.
- The embedded data was used for the scree plot, silhouette scores, PCA and clustering using *Sklearn*'s *KMeans* function.

##### 2. A Hybrid approach to LLM and K-Means:
- The hybrid approach involved embedding the categorical variables, *gender* and *preferred_category*, using the same method as the LLM and K-Means approach. The numeric data however was not covered into text, it was rather normalised using *Sklearn*'s *StandardScaler* function.
- The embeddings of the categorical data and the normalised numeric data were used for the scree plot, silhouette scores, PCA and clustering using *Sklearn*'s *KMeans* function.

##### 3. K-Prototypes:
- The categorical variable's data types were changed from *object* to *category* so that the data would be compatible with *kmodes*'s *KPrototypes* clustering function.
- The numeric variables' were normalised using the *Sklearn*'s *StandardScaler* function.
- The combined data were used for the scree plot, PCA and clustering using *kmodes*'s *KPrototype* function.
- To calculate silhouette scores for the different clusters a Gower dissimilarity matrix was used. The data used to calculated the dissimilarity were the numeric variables that had been normalised using *Sklearn*'s *StandardScaler* and the categorical variables that had been transformed using One-Hot Encoding.

### Model Selection:
See the table below for a summary of how each clustering approach performed.

| Approach | Clusters | Statistical Performance | Cluster Analysis |
| :-------------- | :-------------- | :-------------- | :-------------- |
| K-Means | Six | There was no distinct elbow in the scree plot, and the range of clusters had low silhouette scores that varied between 0.098 and 0.108. When visualising the clusters in 3D, there were six clear cluster all grouped together with some overlap. | The clusters were well separated between all categories. |
| LLM and K-Means | Five | There was a distinct elbow at k=5 in the distortion scree plot, and silhouette scores were varied between 0.238 and 0.536, with cluster five having the highest silhouette score. When visualising the clusters in 3D, there were five distinct, tight and separate clusters. | The clusters were not well separated - each cluster appeared to have a similar distribution of all continuous variables and no distinguishable characteristics for the cluster.   |
| LLM and K-Means Hybrid | Five | There was a distinct elbow at k=5 in the distortion scree plot, and silhouette scores were varied between 0.313 and 0.656, with cluster five having the highest silhouette score. When visualising the clusters in 3D, there were five distinct and separate clusters. | The clusters were not well separated - each cluster appeared to have a similar distribution of all continuous variables. The only distinguishable feature of each cluster was the preferred category - each cluster was made up of entirely one category. |
| K-Prototypes | Five | There was no distinct elbow in the scree plot, and the range of clusters had low silhouette scores that varied between 0.056 and 0.081. When visualising the clusters in 3D, there were five clear cluster all grouped together with overlap. | There was good separation of categories between clusters. |


Based on the information in the table above and additional information in the notebooks, the K-Means approach was selected. Even though the K-Means approach preformed statistically poorer than the LLM K-Means and the hybrid LLM K-Means approach because:
- The K-Means approach's silhouette score was much lower indicating the quality of the cluster was worse.
- The 3D visualisation of the LLM K-Means and the hybrid LLM K-Means had distinct, separate and tight cluster.
The quality of information that was provided by the cluster was poor, meaning that there were no distinguishable features in the clusters. This indicates that the relationships between the variables could have been lost during the embedding process and no meaningful clusters for the business were identified but rather latent relationships were identified and clustered.

The K-Means and the K-Prototype approach preformed statistically similarly, with the K-Prototypes approach having slightly lower silhouette scores. However, this was not the deciding factor when selecting the K-Means approach. The K-Means approach was selected based on the information the clustering provided, it separated the gender, age and income into more distinct categories.

In summary, the K-Means approach was selected even though it did not outperform the other approaches statistically but rather offered meaningful customer segments which can be used when creating targeted marketing strategies. The approaches that statistically better than the chosen approach did not produce meaningful cluster but were rather clustered based on latent relationships and would therefore not be of use to the marketing team.

## Results and Cluster Analysis
### Results:
 As seen in Figure 2, there is no clear elbow however at k=6 there is a significant drop in fit time which corresponds to one of the highest silhouette scores in Figure 3. There is no steady increase in the silhouette score after k=6, the silhouette score fluctuates.

<figure>
    <img src="/assets/elbow.png" style="width: 75%; height: auto;">
    <figcaption>Figure 2: Distortion Score Elbow for KMeans Clustering</figcaption>
</figure>
</br>
</br>

<figure>
    <img src="/assets/sil.png" style="width: 75%; height: auto;">
    <figcaption>Figure 3: Silhouette Score Analysis for Optimal K</figcaption>
</figure>
</br>
</br>

Therefore, using the elbow plot, Figure 2, and the silhouette score per cluster, Figure 3, a six cluster approach was selected.

<figure>
    <img src="/assets/sil_vis.png" style="width: 75%; height: auto;">
    <figcaption>Figure 4: Silhouette Plot of KMeans Clustering for 1000 Samples in 6 Centers</figcaption>
</figure>
</br>
</br>

When analysing the individual silhouette scores across each cluster in Figure 4, each cluster's silhouette score is larger than the average silhouette score. This suggests that the clusters are well separated and internally consistent.

However the overall silhouette score is low, indicating that the clusters overlap this is evident when viewing the clusters using PCA in Figure 5.

<figure>
    <img src="/assets/pca_clusters.png" style="width: 75%; height: auto;">
    <figcaption>Figure 5: Visualising the Clusters using Two Principal Components</figcaption>
</figure>
</br>
</br>

The separation of the clusters can be visualised more clearly in Figure 6.
<figure>
    <iframe src="/assets/interactive_plot.html" style="width: 75%; height: auto;" frameborder="0"></iframe>
    <figcaption>Figure 6: Visualising the Clusters using Three Principal Components</figcaption>
</figure>
</br>
</br>

### Cluster Analysis:

