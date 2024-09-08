# Spotify Genre Classification: Pop vs Classical

This project implements a logistic regression model to classify Spotify songs as either Pop or Classical based on their 'liveness' and 'loudness' features.

## Dataset

- The dataset contains 9,386 Pop songs and 9,256 Classical songs.
- Features used: 'liveness' and 'loudness'

## Model

A logistic regression classifier was with the following characteristics:

- Learning rate: 0.0001 (found to be optimal after testing various rates)
- Number of epochs: 60

## Key Findings

1. Data Distribution:
   - Pop songs are highly concentrated in the high loudness range and spread across liveness.
   - Classical songs are spread across loudness but somwhat more concentrated in the low liveness range.

2. Model Performance:
   - Accuracy: 90.90%
   - Precision: 85.89%
   - Recall: 98.27%
   - F1 Score: 91.66%

3. Decision Boundary:
   - The model relies more heavily on loudness than liveness for classification.
   - There's significant overlap between genres near the decision boundary, especially in the mid-range of liveness.

4. Error Analysis:
   - The model is more likely to misclassify Classical songs as Pop (298 cases) than vice versa (32 cases of Pop songs as Classical).
   - This bias is likely due to the greater spread of Classical songs across the feature space.

## Conclusion

The logistic regression model overall performs well in distinguishing between Pop and Classical songs, with particularly high effectiveness in identifying Pop songs correctly.
