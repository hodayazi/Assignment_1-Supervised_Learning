# REPORT — Module 3 · Assignment 1 · Supervised Learning

**Name:** Hodaya Zinowits  **ID:** 321798159  **Date:** 28/06/2026
**Chosen task:** A (A · Olist negative review / B · Telco churn / C · Olist daily volume forecast)

> Keep this report in English. Be concise and honest. "I don't know, and here is why" is a professional answer. Empty confidence is not.

---

## 1. Problem framing
Business question (one short paragraph):
Predict whether an Olist order will receive a negative review (`review_score <= 2`) using order, payment, item, and delivery features. This prediction can identify unhappy customers early so the business can intervene before the relationship weakens or churn occurs.

Target definition:
Binary target `is_negative`: 1 if `review_score <= 2`, otherwise 0.

**Primary metric and why it fits the business cost** (cost of a false positive vs a false negative; for forecasting, cost of over- vs under-forecasting):
F1-score is the primary metric because the target is imbalanced and the business cost is asymmetric. A false negative means missing a genuinely dissatisfied customer and losing the chance to recover them, while a false positive only triggers an extra review or outreach. F1 balances precision and recall, making it a better proxy for operational value than raw accuracy.

---

## 2. Results table
Fill in every model **and the dumb baseline** on the locked test set.

| Model | Primary metric | Secondary metric | Notes |
|---|---|---|---|
| baseline | F1 = 0.000 | ROC-AUC = 0.500 | DummyClassifier predicting the most frequent class |
| linear | F1 = 0.176 | ROC-AUC = 0.676 | Logistic Regression |
| bagging | F1 = 0.338 | ROC-AUC = 0.693 | Random Forest; best F1 on locked test set |
| boosting | F1 = 0.335 | ROC-AUC = 0.711 | Tuned Gradient Boosting; best AUC |

---

## 3. Guiding questions (graded)
Answer each in 2-5 sentences.

1. **Accuracy trap.** Why is plain accuracy misleading for your problem, and what did the confusion matrix (or the error distribution, for forecasting) reveal that accuracy hid?
Plain accuracy is misleading because 84% of orders are non-negative reviews, so a model that predicts every order as non-negative would still appear accurate. The confusion matrix reveals that this trivial strategy completely misses the minority negative-review class, so accuracy hides the model's inability to identify dissatisfied customers. F1, ROC-AUC, and confusion metrics are needed to show actual performance on the business-critical class.

2. **Cost of errors.** What does a false positive cost vs a false negative in your business context, and how did that drive your metric choice?
A false negative means failing to flag an unhappy customer, which can cost customer satisfaction and future purchases. A false positive means engaging a likely satisfied customer unnecessarily, which is less costly operationally. That asymmetry led to F1 as the primary metric, since it values both precision and recall and is more meaningful than pure accuracy in this imbalanced classification.

3. **Worth deploying?** By how much did your best model beat the dumb baseline? If the margin is small, is the model worth shipping — and would you ship it?
The best model improved F1 from 0.000 (baseline) to 0.338, which is a meaningful lift over a dumb majority-class predictor. The margin shows the model is useful, but the absolute F1 remains modest, so I would deploy cautiously in a pilot phase with close monitoring rather than a full production rollout.

4. **What drives it.** Which features carry the predictions? Do they make business sense, or is the model leaning on a leak or a spurious correlation? How did you check?
The strongest signals are delivery-related features such as `delay_days`, `delivery_days`, and order characteristics like `n_items`. These make business sense because delivery performance is a common driver of negative reviews. I avoided leakage by using a time-based split and excluding raw timestamps from feature inputs, so the model is not directly using future information.

5. **Worst errors.** Look at the 5 worst mistakes. What is the story — a data quality problem, mislabeled rows, or genuinely hard cases?
The top five errors were confident false negatives: the model predicted non-negative while the review was actually negative. These cases often had normal or early delivery metrics, suggesting the dissatisfaction may come from product quality, value perception, or issues not captured by the available features. This points to genuinely hard cases rather than a simple data label problem.

6. **Stability.** How much does the score move across CV folds? Would you hand this single number to a stakeholder as "the" performance? Why or why not?
The tuned Gradient Boosting model had cross-validation F1 = 0.5402 ± 0.0138, showing fairly stable training performance. However, the locked test F1 dropped to 0.335, so I would not present a single number as the final answer without also reporting the CV range and monitoring ongoing performance. The gap suggests the model can overfit and that a stakeholder should see a performance band rather than a single point estimate.

7. **Leakage / time.** (Required for B too, in one line.) How did you guarantee the model never saw information from the future or from the test set? For a time-based split, what happens to the score compared with a random split, and why?
I used a time-based split at 2018-05-25 11:03:29, training on past orders and testing on future orders, so the model never sees future information from the test set. A random split would leak future delivery and order patterns into training, inflating metrics and giving an overly optimistic score.

8. **Monday morning.** If this went live today, what would you monitor, and what signal would make you retrain it?
I would monitor the model F1 and ROC-AUC on fresh orders, the rate of negative reviews, and drift in key features like `delay_days`, `delivery_days`, and `price_total`. A sustained drop in F1, a sudden rise in false negatives, or systematic drift in delivery metrics would trigger retraining.

---

## 4. Model Card

1. Overview
- Task / business question: To predict whether a customer will leave a negative review (`is_negative`) on an order, enabling proactive intervention to improve customer satisfaction.
- Dataset (Option A) and time range: Olist e-commerce data (Option A) using a time-based split (80% training, 20% testing). Training data up to `2018-05-25 11:03:29.800000`, testing data after this timestamp.
- Target definition: `is_negative` is a binary classification target, where '1' indicates a review score of 2 or less (negative review), and '0' otherwise.

2. Metric & performance
- Primary metric and WHY: The F1-score was chosen as the primary metric. This is due to the imbalanced nature of the target variable and the business cost implications. A false negative (failing to identify a genuinely unhappy customer) is considered more costly than a false positive (proactively engaging a customer who would not have left a negative review), as it represents a missed opportunity for service recovery and customer retention. The F1-score provides a balanced measure of precision and recall.
- Dumb baseline score: 0.000 F1-score (achieved by `DummyClassifier(strategy="most_frequent")`, which always predicts the majority class, '0').
- Best model score (on the locked test set): The tuned Gradient Boosting Classifier achieved an F1-score of 0.335 and an ROC-AUC score of 0.711 on the locked test set.
- Did it beat the baseline meaningfully? Is it worth deploying? Yes, the model significantly outperforms the baseline. While the absolute F1 is still modest, it demonstrates value and should be deployed carefully with continued monitoring and improvement.

3. What the model relies on
- Top features and whether they make business sense: The most impactful features are `delay_days`, `n_items`, and `delivery_days`. These features make business sense because delayed deliveries and order complexity are directly tied to customer experience and likely negative reviews.
- Any feature you suspect is a leak or spurious? No obvious leak was found; the time-based split and exclusion of raw timestamps avoid future data leakage.

4. Limitations & failure modes
- The 5 worst errors — what is the pattern? The top 5 most confident mistakes were false negatives with low predicted probability for the negative-review class. These cases suggest the remaining sources of dissatisfaction may be outside the available delivery and payment features.
- Where would this model break? The model may fail when customer dissatisfaction is driven by product-specific issues, seller behavior, or changes in buyer expectations not captured by the current feature set. It may also degrade under major logistics disruptions or if the review distribution shifts over time.
- Stability across folds (mean +/- std): Gradient Boosting F1 = 0.5402 ± 0.0138 across 5 stratified CV folds.

5. Fairness / ethics
- Could any group be systematically mis-served by this model? Yes. The model's reliance on delivery features could disadvantage customers in remote or higher-cost shipping regions, who may be more likely to receive late or expensive deliveries. If interventions are based on these predictions, fairness across geographic and seller segments should be monitored.

6. Real world
- If deployed Monday: what would you monitor? I would monitor the negative review rate, model F1 and ROC-AUC, false negative volume, and drift in delivery-related features such as `delay_days`.
- What signal would make you retrain it? A sustained drop in F1, a sudden increase in false negatives, or significant drift in key feature distributions would trigger retraining.
```

---

## 5. Reflection
What surprised you? What would you do differently with two more weeks or more data? How would this approach change if it were part of your mid-term project?
I was surprised that the dumb baseline still scored 0.000 F1, which highlights how much harder the problem is than raw accuracy suggests. With two more weeks or more data, I would explore richer customer and product features, handle class imbalance more aggressively, and test additional model families or ensemble combinations. For a mid-term project, I would also build a monitoring pipeline and connect the prediction output to a concrete intervention workflow so the model could be evaluated by actual business impact.
