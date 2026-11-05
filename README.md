# Late Deliveries

QuickBite wants to know, when an order is placed, whether it will arrive more than
ten minutes late, so it can offer a $4 voucher first. The model predicts the
probability that an order is late; a cost-based threshold turns it into a decision.

## Setup

```
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```
python -m latedeliveries.train
```

## Check it

```
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

**Columns used.** `distance_km`, `items`, `hour`, `prep_minutes`, `courier_months`
(numeric) and `weather`, `day_type`, `vehicle` (category). All are known, or
estimated by the kitchen, when the order is placed.

**Columns left out.**
- `customer_rating`: customers give it after the delivery, so it does not exist at
  prediction time. It also depends on the outcome (late orders get lower ratings).
- `order_id`: only a label, with no link to lateness.
- `late`: the target.

*Leakage check:* adding `customer_rating` back raised cross-validated ROC-AUC from
0.795 to 0.907 (+0.112). A jump that large from one column is a warning of
leakage, not an improvement, so the column stays out.

**Split.** Stratified 75/25 (2,250 train, 750 test), `random_state=42` held in one
constant. The test set is untouched until the final scoring.

**Choosing the model.** Five-fold stratified CV on the training data only:

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| baseline (dummy) | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| logistic regression | 0.862 ± 0.008 | 0.654 ± 0.065 | 0.242 ± 0.053 | 0.352 ± 0.064 | 0.795 ± 0.034 |
| logistic (balanced) | 0.737 ± 0.019 | 0.333 ± 0.027 | 0.681 ± 0.049 | 0.447 ± 0.034 | 0.795 ± 0.034 |
| random forest | 0.859 ± 0.008 | 0.736 ± 0.121 | 0.148 ± 0.036 | 0.246 ± 0.056 | 0.786 ± 0.037 |
| hist. gradient boosting | 0.851 ± 0.006 | 0.544 ± 0.036 | 0.288 ± 0.037 | 0.375 ± 0.036 | 0.754 ± 0.024 |

Accuracy is nearly useless here: the baseline that flags nobody scores 0.844. The
real question is cost. Using the same method as the threshold step (out-of-fold
probabilities, best threshold on the 0.05 grid), the lowest training cost for each
model was: logistic regression $4,031, balanced logistic $4,098, random forest
$4,199, gradient boosting $4,327. Flagging nobody costs $5,265 on the same data.
I chose plain **logistic regression**: it is cheapest, its ROC-AUC is tied with the
best, and its probabilities are not distorted by class weights, which matters
because the threshold is read straight off the cost rule. The differences between
the top models are within the fold-to-fold spread, so simplicity decides.

**Tuning.** `GridSearchCV` over `model__C` (7 values) and `model__class_weight`
(None, balanced): 14 combinations, scored by ROC-AUC because it judges the ranking
of orders independently of any threshold, and the threshold is chosen later from
costs. Best: `C=3`, `class_weight=None`, ROC-AUC 0.7953 against 0.7951 untuned. The
gain (0.0002) is far smaller than the fold-to-fold spread (0.034), so **tuning did
not beat the untuned model in any meaningful way.**

**Threshold.** An order is flagged when its probability is at least the threshold.
A voucher is worth it when p x $15 > $4, i.e. p > 0.27, so the best threshold should
sit near there. Out-of-fold probabilities from the training data (`cross_val_predict`)
were scored at 0.05 to 0.95 in steps of 0.05, and the cheapest was **0.30** (ties go
to the lower threshold). A threshold is needed because the cost of a miss ($15) is
nearly four times the cost of a voucher ($4), so the default 0.5 would miss too
many late orders.

## Result

Test set (750 orders, 117 late), scored once at threshold 0.30:

| | not flagged | flagged |
|---|---|---|
| on time | 578 | 55 |
| late | 64 | 53 |

Precision 0.491, recall 0.453, F1 0.471, ROC-AUC 0.796.

| policy | total cost |
|---|---|
| flag nobody | $1,755 |
| flag everybody | $3,000 |
| model at 0.30 | $1,392 |

The model saves $363 (about 21%) over the current policy of flagging nobody.
With only 117 late orders in the test set and a fold-to-fold ROC-AUC spread of
0.034, I would trust that the model helps, but not the exact size of the saving.

## What the tests cover

- the feature list excludes `order_id`, `customer_rating` and `late` (parametrized)
- the split is 75/25, stratified and repeatable
- predicted probabilities are valid and the pipeline handles a missing
  `prep_minutes` and an unseen weather value
- the cost function and threshold choice on small hand-worked examples, including
  cases where flagging nobody and flagging everybody are cheapest
- a saved model predicts the same after reloading

Not tested: `train.main()` end to end. It takes a while to run and writes into the
repository. I check it instead by running it on a fresh clone and comparing
`reports/metrics.json` with the committed copy.

## Known limitations

- The data are invented, and the $4 and $15 costs are fixed assumptions.
- The threshold grid is coarse (0.05) and chosen from one set of folds.
- No calibration check on the probabilities.
- Only one tuned model; no cost-based scoring in the grid search.

