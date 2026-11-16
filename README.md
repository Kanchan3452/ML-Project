# ML-Project
1. Understanding the project
What is your project about?
Your project analyses the district-wise distribution of LPG connections released under the Pradhan Mantri Ujjwala Yojana (PMUY) in West Bengal.

You have data for 23 districts, covering financial years 2016–17 to 2022–23 (April–February).

You use Machine Learning to:
1. Study changes in LPG connection releases over time.
2. Predict connection releases using regression models.
3. Group districts with similar patterns using clustering.
4. Analyse clean-fuel usage using NFHS-5 data.
What is PMUY?
Pradhan Mantri Ujjwala Yojana (PMUY) is a Government of India scheme launched in 2016 to provide LPG connections to eligible women from low-income households.

Its main purpose is to encourage the use of cleaner cooking fuel instead of traditional fuels such as wood, coal and dung cakes.
Why did you choose this project?
Because it combines a real-world social issue with Machine Learning.

Instead of using an artificial dataset, you analyse government-related data to understand patterns in LPG connection distribution and clean-fuel usage.

2. What is Machine Learning?
Machine Learning is a branch of Artificial Intelligence in which computers learn patterns from data and use those patterns to make predictions or decisions.

In traditional programming, we provide the computer with rules and data to get an output.

In Machine Learning, we provide data and allow the algorithm to learn patterns from it.
Types of Machine Learning
There are three major types:
1. Supervised Learning
Uses labelled data
The model learns from data where the correct answers are already known.

Example: Predicting the number of LPG connections using previous years' data.

2. Unsupervised Learning
Uses unlabelled data
The model finds hidden patterns or groups in the data without being given predefined answers.

Example: Grouping West Bengal districts based on their LPG connection patterns.

3. Reinforcement Learning
Uses rewards and penalties
An agent learns by taking actions and receiving rewards or penalties.

Example: A robot learning how to navigate an environment.

Which types did you use?
- Supervised Learning: Regression algorithms.
- Unsupervised Learning: Clustering algorithms.
- Reinforcement Learning: Not used.
3. Understanding Regression
Regression is a supervised Machine Learning technique used to predict a numerical value.

For example:
- Predicting house prices.
- Predicting temperature.
- Predicting sales.
- Predicting LPG connection releases.
In your project:
- Input: Year and district information.
- Output: Number of PMUY connections released.
You used three regression models in the updated pooled comparison.
3.1 Linear Regression










What is Linear Regression?

Linear Regression is an algorithm that finds a straight-line relationship between input and output variables.

It tries to find the line that minimises the squared differences between actual and predicted values.

The equation is:
\[
Y = mX+c
\]

Where:
- \(Y\) = Predicted value.
- \(X\) = Input variable.
- \(m\) = Slope of the line.
- \(c\) = Intercept.
How does it work in your project?

The model studies how the number of PMUY connections changes with time.

It tries to identify a general increasing or decreasing trend.

Why did we use Linear Regression?
- It is simple to understand.
- It is easy to implement.
- It provides a basic trend for comparison.
- It works well when the relationship is approximately linear.
Why not use only Linear Regression?

Because the connection-release pattern may not follow a straight line. It may increase, decrease or change direction over time.

Therefore, we also tested Polynomial Regression.
3.2 Polynomial Regression










What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression that can represent curved relationships.

For degree two, its equation is:
\[
Y = b_0+b_1X+b_2X^2
\]

Here, \(X^2\) allows the model to represent a curved trend.

How does it work in your project?

It examines whether the annual PMUY connection pattern follows a curved trend rather than a straight-line trend.

Why did we use it?
- To capture possible changes in the direction of a trend.
- To compare a more flexible model with Linear Regression.
- To understand whether a curved relationship fits the data better.
Why not use a higher degree, such as 3 or 4?

Higher-degree polynomial models can become too complex.

They may fit the training data very closely but perform poorly on new data. This is called overfitting.

Since your dataset contains only six complete years, a degree-two polynomial is a simpler choice to test.
3.3 Ridge Regression










What is Ridge Regression?

Ridge Regression is a modified form of Linear Regression that uses regularization.

Regularization means adding a penalty to the model to control very large coefficients.

It uses L2 regularization.

Its objective function is:
\[
\text{Loss}=\text{SSE}+\alpha\sum_{j=1}^{n}\beta_j^2
\]

Where:
- SSE = Sum of squared prediction errors.
- \(\alpha\) = Regularization strength.
- \(\beta_j\) = Model coefficients.
How does it work in your project?

Ridge Regression uses district information and the year index to predict PMUY connections.

Your updated model uses alpha = 1.0.

Why did we use Ridge Regression?
- It controls large coefficients.
- It helps reduce model complexity.
- It can reduce overfitting.
- It is useful when a model has several input features.
Why Ridge instead of Lasso Regression?

Both are regularization techniques.
Ridge Regression	Lasso Regression
Uses L2 penalty	Uses L1 penalty
Shrinks coefficients	Can make some coefficients exactly zero
Generally keeps all features	Can remove less useful features

We used Ridge as the regularized comparison model. Lasso was not part of the implemented model comparison.
Important: What is the difference between the three?
Algorithm	Main purpose
Linear Regression	Fits a straight-line trend
Polynomial Regression	Fits a curved trend
Ridge Regression	Controls large coefficients using regularization


4. Understanding Pooled Regression
This is one of the most important parts of your updated project.

What does pooled mean?

Pooled means combining data from multiple districts into one dataset instead of training a separate prediction model for every district.

Your project has:
- 23 districts.
- 6 complete financial years.
- 138 district-year observations.
\[
23\times6=138
\]

Each row represents one district in one financial year.
Why did we use pooled regression?
The earlier approach trained separate models for individual districts. Each model had only six yearly observations.

That is a very small amount of data.

The pooled approach combines all districts so that the model can learn from more observations.
How does the model identify districts?
We use One-Hot Encoding.

What is One-Hot Encoding?

It converts categorical information into numerical columns.

For example:
District	Kolkata	Bankura	Darjeeling
Kolkata	1	0	0
Bankura	0	1	0
Darjeeling	0	0	1

Here:
- 1 means the district matches that column.
- 0 means it does not.
Why is it needed?

Machine Learning algorithms generally work with numerical input. One-Hot Encoding converts district names into a numerical representation.

Why not simply assign Kolkata = 1, Bankura = 2 and Darjeeling = 3?

Because that would create an artificial numerical order between districts. One-Hot Encoding avoids implying that one district is numerically greater than another.
5. Training and Testing
What is training?

Training means providing data to the model so that it can learn patterns.

What is testing?

Testing means giving the trained model data it has not seen before and checking how well it predicts.

For example:
- Training data: Previous years.
- Testing data: A later year.
Why do we need both?
A model might perform very well on the data it has already seen but poorly on new data.

Testing helps us understand whether the model has learned useful patterns or simply memorised the training data.
5.1 Chronological Validation
Your project's chronological testing approach

The model always learns from earlier years before predicting a later year.
Test 1

Training: 2016–17 to 2018–19

Testing: 2019–20

Test 2

Training: 2016–17 to 2019–20

Testing: 2020–21

Test 3

Training: 2016–17 to 2020–21

Testing: 2021–22

Total: 69 held-out observations per model (23 districts × 3 test years).

Why did we use chronological validation instead of random splitting?

Because this is time-related data.

If we randomly split the years, the model might train on a future year and then be tested on an earlier year.

That would not represent a realistic forecasting situation.

Chronological validation ensures that we predict the future using the past.
6. Understanding Model Evaluation Metrics
These metrics help us understand whether our models are performing well.
6.1 R² Score
R² stands for Coefficient of Determination.

It measures how well the model's predictions compare with the actual values.
- R² = 1: Perfect predictions.
- R² = 0: Same performance as predicting the test-set average.
- R² less than 0: Worse than predicting the test-set average.
Why is your R² negative?

Your models are struggling to predict unseen years accurately.

One important reason is that the dataset has only six complete years and includes major changes in annual connection releases.

A negative R² does not mean your code is incorrect.
6.2 MAE
MAE stands for Mean Absolute Error.

It calculates the average absolute difference between actual and predicted values.
\[
MAE=\frac{1}{n}\sum |y_i-\hat y_i|
\]

Simple example:

Actual connections = 100

Predicted connections = 80

Absolute error = 20

A lower MAE means smaller average errors.
6.3 RMSE
RMSE stands for Root Mean Squared Error.

It measures prediction errors but gives more importance to larger errors.
\[
RMSE=\sqrt{\frac{\sum(y_i-\hat y_i)^2}{n}}
\]

Why use RMSE along with MAE?

MAE treats errors proportionally, whereas RMSE penalises large mistakes more heavily.

Using both helps us understand prediction errors from different perspectives.
Difference between MAE and RMSE
MAE	RMSE
Average absolute error	Square root of average squared error
Less affected by large errors	More affected by large errors
Easy to interpret	Highlights large mistakes


7. Understanding Clustering
Clustering is an unsupervised Machine Learning technique.

It groups similar data points together without providing predefined labels.

In your project, clustering is used to group districts according to their PMUY connection-release patterns.

You used two algorithms.
7.1 K-Means Clustering










What is K-Means?

K-Means divides data into K groups based on similarity.

Each group is called a cluster.

How does K-Means work?
1. Select the number of clusters (K).
2. The algorithm places initial cluster centres called centroids.
3. Each data point is assigned to its nearest centroid.
4. The centroids are recalculated.
5. These steps repeat until the clusters stop changing significantly.
What is a centroid?

A centroid is the centre point of a cluster. It represents the average position of the observations in that cluster.

How did you use it?
- Each district is represented using six yearly PMUY values.
- The values are standardised.
- Different K values from 2 to 5 are tested.
- Silhouette Score is used to compare the clusters.
Your selected K-Means model uses two clusters, with a silhouette score of 0.432.

Why did we use K-Means?
- It is simple and easy to implement.
- It works well for grouping numerical data.
- It is computationally efficient.
- It helps identify similarities between districts.
Why not use K-Means with many clusters?

Too many clusters can divide districts into very small groups and make the results difficult to interpret. We compared several values of K and selected K=2 based on the evaluated silhouette scores.
7.2 Hierarchical Clustering










What is Hierarchical Clustering?

Hierarchical Clustering creates a hierarchy of groups based on the similarity between data points.

Your project uses Agglomerative Hierarchical Clustering.

What is Agglomerative Clustering?

It is a bottom-up approach.

It starts by treating each district as an individual cluster and gradually merges the most similar clusters.

How does it work?
1. Each district starts in its own cluster.
2. The algorithm calculates similarities or distances.
3. The closest clusters are merged.
4. The process continues until the desired grouping is reached.
What is a dendrogram?

A dendrogram is a tree-like diagram that shows how clusters are joined together.

It helps us understand the grouping structure.

What is Ward's Linkage?

Ward's linkage merges clusters in a way that minimises the increase in within-cluster variation.

In simple words, it tries to keep the districts inside each cluster as similar as possible.

Why did we use Hierarchical Clustering?
- It shows how groups are formed.
- It provides a visual representation through a dendrogram.
- It allows us to compare its results with K-Means.
Why not use only K-Means?

Because Hierarchical Clustering gives a different view of the data. Comparing both helps us understand whether the grouping patterns are similar.
Difference between K-Means and Hierarchical Clustering
K-Means	Hierarchical
Uses centroids	Uses cluster distances
Requires K in advance	Builds a hierarchy
Repeatedly updates cluster centres	Gradually merges clusters
Does not naturally produce a dendrogram	Can produce a dendrogram

8. What is Standardisation?
Standardisation is a data-preprocessing technique that brings numerical features to a comparable scale.

It uses the formula:
\[
Z=\frac{X-\mu}{\sigma}
\]

Where:
- \(X\) = Original value.
- \(\mu\) = Mean.
- \(\sigma\) = Standard deviation.
After standardisation, the feature has a mean close to 0 and standard deviation close to 1.

Why did we use it?

K-Means and Hierarchical Clustering use distances to measure similarity.

If one feature has much larger numerical values than another, it may dominate the distance calculation.

Standardisation helps prevent that.

Why not use Min-Max Normalisation?

Min-Max scaling converts values into a fixed range, usually 0 to 1.

Standardisation was selected for the clustering workflow because it centres the features and scales them according to their variation. Both methods are useful, but they work differently.
9. Understanding NFHS-5 Analysis
What is NFHS-5?

National Family Health Survey (NFHS-5) is a nationwide survey conducted during 2019–21.

It collects information about health, nutrition, family welfare and household conditions.

Your project uses its clean-fuel indicator.

What does clean fuel mean?

Clean cooking fuels include LPG, electricity, biogas and other fuels classified as clean under the survey's indicator.

Why did we include NFHS-5?

PMUY data tells us how many new LPG connections were released.

NFHS-5 gives us information about the percentage of households using clean cooking fuel.

Combining these datasets provides a broader view of clean-cooking access.

Why are three districts missing NFHS-5 values?

The compiled dataset does not have values for Alipurduar, Jhargram and Kalimpong. We keep them missing rather than treating them as zero.

Important: PMUY connection releases and NFHS clean-fuel usage are different measurements. We cannot conclude that one directly caused the other from this analysis.

10. Understanding Your Actual Results
These are the main results you should remember for your viva.
Regression comparison
Model	Test R²
Pooled Linear	-2.3446
Pooled Polynomial	-6.6962
Pooled Ridge	-2.0654

Best relative model: Ridge Regression
It has the least negative R² and the lowest MAE and RMSE.

Clustering comparison
Method	Silhouette
K-Means	0.432
Hierarchical	0.407

Better separation: K-Means

NFHS-5 results
- Available districts: 20
- Average district percentage: 39.3%
- Highest: Kolkata – 91.7%
- Lowest: Purba Medinipur – 17.0%

11. Important Viva Questions and Answers
Try to understand these answers instead of memorising them word for word.

Q1. Why did you choose this project?

I wanted to apply Machine Learning to a real-world problem. This project helped me analyse government data and understand the distribution of LPG connections across districts.

Q2. Which Machine Learning algorithms did you use?

I used Linear Regression, Polynomial Regression, Ridge Regression, K-Means Clustering and Hierarchical Clustering.

Q3. Which algorithms are supervised and which are unsupervised?

Linear, Polynomial and Ridge Regression are supervised algorithms. K-Means and Hierarchical Clustering are unsupervised algorithms.

Q4. Why did you use three regression algorithms?

To compare different approaches. Linear Regression captures straight-line trends, Polynomial Regression captures curved trends and Ridge Regression controls large coefficients through regularisation.

Q5. Which regression model performed best?

Ridge Regression performed best relative to the other two models, although its test R² was negative.

Q6. What does negative R² mean?

It means the model performed worse than simply predicting the average value of the test data.

Q7. Does negative R² mean your project failed?

No. The code ran successfully and the evaluation gives us an honest picture of the model's limitations. A model does not need to have a positive R² for an experiment to be valid.

Q8. Why did you use chronological validation?

Because our data is time-based. We should use past years to predict future years, not randomly mix future and past observations.

Q9. Why did you use pooled regression?

Because training separate models for each district gave us only six observations per model. Pooled regression combines district-year observations, giving us 138 observations in total.

Q10. Why did you use K-Means?

To group districts with similar PMUY connection-release patterns without predefined labels.

Q11. Why did you use two clusters?

We tested different cluster counts from 2 to 5 and selected two based on the silhouette-score comparison.

Q12. What is a silhouette score?

It measures how well-separated the clusters are. A higher score generally indicates better clustering.

Q13. Why did you use Hierarchical Clustering?

To compare another clustering approach and understand how districts can be grouped using a dendrogram.

Q14. What is overfitting?

Overfitting happens when a model learns the training data too closely, including noise, and performs poorly on new data.

Q15. What is the difference between MAE and RMSE?

Both measure prediction errors. RMSE gives more importance to large errors, while MAE treats errors proportionally.

Q16. Why did you use NFHS-5 data?

To study the percentage of households using clean cooking fuel and compare the broader clean-fuel situation across districts.

Q17. What are the limitations of your project?

The dataset contains only six complete years for regression, three districts have missing NFHS-5 values, and all three models have negative test R².

Q18. What can be improved in the future?

We can add more years of data, LPG refill information, household income and rural-urban information to improve the analysis.
