# Customer Segmentation for Marketing Analysis

## Data
The dataset for this project can be found on [Kaggle](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis) under the CC0: Public Domain license.

The dataset consists of simulated customer data, containing demographic and behavioural information for 1 000 customer records. 
The data is unlabelled, therefore unsupervised machine learning methods will be used to discover patterns and relationships in the data.

The dataset is mixed - it is made up of numeric and nominal categorical variables:
- **id:** Unique customer identifier.
- **age:** Customer age.
- **gender:** Customer gender (Female, Male, Other).
- **income:** Annual income in USD.
- **spending_score:** Customer's spending behaviour and loyalty (1-100).
- **membership_years:** Years customer has been a member.
- **purchase_frequency:** Number of purchases made by the customer in the late year.
- **preferred_category:** Preferred shopping category (Electronics, Clothing, Groceries, Home & Garden, Sports).
- **last_purchase_amount:** Amount spent in USD by the customer on their last purchase.

## Objectives
The main objective of this project is:
> To identify different customer segments based on the customer's demographic and behavioural information. The customer segments can then be used to develop targeted marketing strategies for each segment

To achieve this objective, it was further broken down into the following four technical sub-objectives:
1. Perform an exploratory data analysis.
2. Perform data cleaning.
3. Determine which unsupervised machine learning method should be used to group the customers into different segments based on their demographic and behavioural information.
4. Provide summaries for each customer segment to assist in developing targeted marketing strategies.

## Main Insights
From the exploratory data analysis, we found that:
- The continuous variables, i.e. *age*, *income*, *spending_score*, *last_purchase_amount* and *purchase_frequency*, appear to be somewhat evenly distributed, see in Figure 1 below.
<figure><p align="center"><img src="/assets/cont_var_dist.png"><br><figcaption>Figure 1: Distribution of Continuous Variables</figcaption></p></figure>
</br>
</br>

- There are no strong relationships between any of the continuous variables mentioned previously. All correlations range between -0.06 and 0.06.
- There are no strong relationships between the continuous variables mentioned above and the categorical variables, *gender* and *preferred_category*. However, there seems to be a slight tendency for males to purchase more frequently and that females last purchase amount tends to be lower than males or others.
- There is an even distribution of preferred shopping categories amongst the different genders.

## Data Cleaning and Engineered Features
The following steps were taken to prepare the data for model selection:
- The dataset was checked for any missing values and duplicated customer data. No missing values or duplicate records were found.
- The continuous variables were normalised using *Sklearn*'s *StandardScaler* function. The Standard Scaler function uses z-score normalisation by transforming the data so that it has a mean of zero and standard deviation of one.
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
3. A Hybrid approach using LLM and K-Means
4. K-Prototypes 

### Methodology:
The following methodology was used for all approaches:
- The whole dataset was used to model the data, as there was no need to create a train-test split.
- The optimum number of clusters was determined by:
  - Finding the elbow or determining the most optimal number of clusters using the distortion score scree plot.
  - Calculating the silhouette score for a range of different clusters.
  - If there was no clear decision as to the optimal number of clusters in the scree plot, i.e. there was no elbow, and there is small fluctuation between the clusters' silhouette scores, then consider using silhouette plots for a range of different clusters. The aim would be to find the smallest cluster size where cluster are not imbalanced and all individual cluster's silhouette coefficients are larger than the average silhouette score, to avoid poorly separated clusters and incorrectly classified data.
- Analysing the silhouette plot for the chosen number of clusters to ensure that each cluster silhouette score is larger than the average silhouette score, there are not many or any incorrectly classified data points and that the cluster sizes appear balanced.
- Cluster the data using the chosen clustering approach using random state 42.
- Perform a principal component analysis using three principal components. Plot 2D and 3D scatterplots coloured by cluster to visualise the clustering quality.
- Preform cluster analysis on the original data using the computed cluster labels. 

#### Additional Methodology for the Different Clustering Approaches:
##### 1. LLM and K-Means:
- The original data is converted into text after combining all variables for each customer, creating a 'customer profile'.
- The text is embedding using *sentence_transformers*'s *SentenceTransformer* function with the *paraphrase-MiniLM-L6-v2* model. The embeddings were normalised by the *SentenceTransformer* function.
- The embedded data was used for the scree plot, silhouette scores, PCA and clustering using *Sklearn*'s *KMeans* function.

##### 2. A Hybrid approach to LLM and K-Means:
- The hybrid approach involved embedding the categorical variables, *gender* and *preferred_category*, using the same method as the LLM and K-Means approach. The numeric data however was not covered into text, but was rather normalised using *Sklearn*'s *StandardScaler* function.
- The embeddings of the categorical data and the normalised numeric data were used for the scree plot, silhouette scores, PCA and clustering using *Sklearn*'s *KMeans* function.

##### 3. K-Prototypes:
- The categorical variable's data types were changed from *object* to *category* so that the data would be compatible with *kmodes*'s *KPrototypes* clustering function.
- The numeric variables' were normalised using the *Sklearn*'s *StandardScaler* function.
- The combined data was used for the scree plot, PCA and clustering using *kmodes*'s *KPrototype* function.
- To calculate the silhouette scores for the different clusters a Gower dissimilarity matrix was used. The data used to calculate the dissimilarity included the numeric variables that had been normalised using *Sklearn*'s *StandardScaler* and the categorical variables that had been transformed using One-Hot Encoding.

### Model Selection:
See the table below for a summary of how each clustering approach performed.

| Approach | Clusters | Statistical Performance | Cluster Analysis |
| :-------------- | :-------------- | :-------------- | :-------------- |
| K-Means | Six | There was no distinct elbow in the scree plot, and all clusters had low silhouette scores that varied between 0.098 and 0.108. When visualising the clusters in 3D, there were six clear clusters all grouped together with some overlap. | The clusters were well separated between all variables. |
| LLM and K-Means | Five | There was a distinct elbow at k=5 in the distortion scree plot, and silhouette scores varied between 0.238 and 0.536, with k=5 having the highest silhouette score. When visualising the clusters in 3D, there were five distinct, tight and separate clusters. | The clusters were not well separated - each cluster appeared to have a similar distribution of all continuous variables and there were no distinguishable characteristics for any cluster.   |
| LLM and K-Means Hybrid | Five | There was a distinct elbow at k=5 in the distortion scree plot, and silhouette scores varied between 0.313 and 0.656, with k=5 having the highest silhouette score. When visualising the clusters in 3D, there were five distinct and separate clusters. | The clusters were not well separated - each cluster appeared to have a similar distribution of all continuous variables. The only distinguishable feature of each cluster was the preferred category - each cluster was made up of entirely one category. |
| K-Prototypes | Five | There was no distinct elbow in the scree plot, and all clusters had low silhouette scores that varied between 0.056 and 0.081. When visualising the clusters in 3D, there were five clear clusters all grouped together with overlap. | There was good separation of variables between clusters. |


Based on the information in the table above and additional information in the notebooks, the K-Means approach was selected. Even though the K-Means approach preformed statistically poorer than the LLM K-Means and the hybrid LLM K-Means approach because:
- The K-Means approach's silhouette score was much lower indicating the quality of the cluster was worse.
- The 3D visualisation of the LLM K-Means and the hybrid LLM K-Means had distinct, separate and tight cluster.

The quality of information that was provided by each cluster was poor, meaning that there were no distinguishable features in the clusters. This indicated that the relationships between the variables could have been lost during the embedding process and no meaningful clusters for creating marketing strategies were identified but rather latent relationships were identified and clustered.

The K-Means and the K-Prototype approach preformed statistically similarly, with the K-Prototypes approach having slightly lower silhouette scores. However, this was not the deciding factor when selecting the K-Means approach. The K-Means approach was selected based on the information the clustering provided, it separated gender, age and income into more distinct categories.

In summary, the K-Means approach was selected even though it did not outperform the other approaches statistically but rather offered meaningful customer segments which can be used when creating targeted marketing strategies. The approaches that performed better statistically than the chosen approach did not produce meaningful clusters but were rather clustered based on latent relationships and would therefore not be of use to the marketing team.

## Results and Cluster Analysis
### Results:
As seen in Figure 2, there is no clear elbow however at k=6 there is a significant drop in fit time which corresponds to one of the highest silhouette scores in Figure 3. In Figure 3 there is no steady increase in the silhouette score after k=6, the silhouette score fluctuates.

<figure>
    <p align="center"><img src="/assets/elbow.png" style="width: 75%; height: auto;"></p>
    <figcaption>Figure 2: Distortion Score Elbow for KMeans Clustering</figcaption>
</figure>
</br>
</br>

<figure>
    <p align="center"><img src="/assets/sil.png" style="width: 75%; height: auto;"></p>
    <figcaption>Figure 3: Silhouette Score Analysis for Optimal K</figcaption>
</figure>
</br>
</br>

Therefore, using the scree plot, Figure 2, and the silhouette score per cluster, Figure 3, a six cluster approach was selected.

<figure>
    <p align="center"><img src="/assets/sil_vis.png" style="width: 75%; height: auto;"><p/>
    <figcaption>Figure 4: Silhouette Plot of KMeans Clustering for 1000 Samples in 6 Centers</figcaption>
</figure>
</br>
</br>

When analysing the individual silhouette scores across each cluster in Figure 4, each cluster's silhouette score is larger than the average silhouette score. This suggests that the clusters are well separated and internally consistent.

However the overall silhouette score is low indicating that the clusters overlap or are close together, this is evident when viewing the clusters using PCA in Figure 5 and 6.

<figure>
    <p align="center"><img src="/assets/pca_clusters.png" style="width: 75%; height: auto;"></p>
    <figcaption>Figure 5: Visualising the Clusters using Two Principal Components</figcaption>
</figure>
</br>
</br>

The separation of the clusters can be visualised more clearly in [Figure 6](assets/interactive_plot.html).
<p align="center">
  <img src="/assets/pca3.png" alt="Figure 6" style="width: 75%; margin-right: 0%;">
  <img src="/assets/pca3_1.png" alt="Figure 7" style="width: 75%; margin-left: 0%;">
</p>

<p align="center">
  <figcaption>Figure 6: Visualising the Clusters using Three Principal Components</figcaption>
</p>

### Cluster Analysis:
#### Cluster 1:
**Average Age:** 33

**Gender:** Female

**Average Income:** $77 992

**Average Spending Score:** 76

**Average Membership Years:** 5.8

**Average Purchase Frequency:** 26

**Preferred Category:** Electronics

**Average Last Purchase Amount:** $234

**Recommendations:** Focus on electronics and Home & Garden products tailored for this segment, focus on new products and implement loyalty programs. 
</br>
</br>

#### Cluster 2:
**Average Age:** 46

**Gender:** Male

**Average Income:** $68 704

**Average Spending Score:** 23

**Average Membership Years:** 7.5

**Average Purchase Frequency:** 27

**Preferred Category:** Sports

**Average Last Purchase Amount:** $337

**Recommendations:** Emphasise sports & fitness products and electronics tailored for this sector, and promote sporting events selling these products.
</br>
</br>

#### Cluster 3:
**Average Age:** 54

**Gender:** Other

**Average Income:** $58 868

**Average Spending Score:** 60

**Average Membership Years:** 4.1

**Average Purchase Frequency:** 27

**Preferred Category:** Sports

**Average Last Purchase Amount:** $743

**Recommendations:** Focus on sport & fitness products, increase personalised recommendations, and implement loyalty programs.
</br>
</br>

#### Cluster 4:
**Average Age:** 30

**Gender:** Male

**Average Income:** $ 99 447

**Average Spending Score:** 33

**Average Membership Years:** 3.4

**Average Purchase Frequency:** 24

**Preferred Category:** Electronics

**Average Last Purchase Amount:** $630

**Recommendations:** Focus on electronic products tailored to this segment, offer personalised recommendations, and strategies on how to increase customer loyalty and purchase frequency.
</br>
</br>

#### Cluster 5:
**Average Age:** 57

**Gender:** Male

**Average Income:** $122 383

**Average Spending Score:** 53

**Average Membership Years:** 3.9

**Average Purchase Frequency:** 22

**Preferred Category:** Sports

**Average Last Purchase Amount:** $314

**Recommendations:** Emphasise sports & fitness products and electronics tailored to this segment, promote sporting events, and introduce subscription services.
</br>
</br>

#### Cluster 6:
**Average Age:** 45

**Gender:** Other

**Average Income:** $109 792

**Average Spending Score:** 62

**Average Membership Years:** 8.5

**Average Purchase Frequency:** 34

**Preferred Category:** Groceries

**Average Last Purchase Amount:** $715

**Recommendations:** Focus on groceries, implement loyalty programs based on membership years and purchase frequency, offer personalised recommendations and promote newer products.
</br>
</br>

See a more detailed breakdown of the categorical and continuous variable distribution amongst clusters in Figure 7 and 8 below.

<figure>
    <p align="center"><img src="/assets/cluster_cont.png" style="width: 100%; height: auto;"></p>
    <figcaption>Figure 7: Distribution of Continuous Variables per Cluster</figcaption>
</figure>
</br>
</br>

<figure>
    <p align="center"><img src="/assets/cluster_cat_breakdown.png" style="width: 100%; height: auto;"></p>
    <figcaption>Figure 8: Distribution of Categorical Variables per Cluster</figcaption>
</figure>
</br>
</br>

