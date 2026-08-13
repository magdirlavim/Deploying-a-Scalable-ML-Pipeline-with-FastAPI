# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a Random Forest Classifier that predicts if someone makes more or less than $50K a year, using Census data.

## Intended Use

This is meant for learning purposes, showing how to build and deploy a machine learning pipeline. It should not be used for real decisions like hiring or loans.

## Training Data

The data is the 1994 US Census dataset, also known as the "Adult" dataset. It was split 80/20 into training and test sets. 

## Evaluation Data

The remaining 20% test split, using the same encoding as training.

## Metrics

_Please include the metrics used and your model's performance on those metrics._
The model was evaluated using Precision, Recall, and F1 score. Model states: 
- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

Performance also varied across different groups (like education level), which is shown in `slice_output.txt`.

## Ethical Considerations

The model may perform differently across race, sex, and other demographic groups, which could lead to unfair predictions if used in real life. 

## Caveats and Recommendations

This model should not be used for real decisions without more testing for fairness. The data is also old, so it may not reflect today's population well.