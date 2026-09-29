# StockSense - Statistical Tests

Alpha = 0.05 for all tests.


## Test 1 - Do promotions significantly change units sold?
- H0: distribution of units_sold is the same with and without promotion.
- H1: distributions differ.
- Test: Mann-Whitney U.
- n(promo) = 36,108, n(normal) = 36,992
- U = 677,301,388.5, p = 0.0009259
- Effect size: Cohen's d = 0.016, rank-biserial = -0.014
- Decision: REJECT H0
- Business reading: promotions raise per-day units sold significantly.

## Test 2 - Does mean demand differ across store types?
- H0: demand distribution is identical across store types.
- H1: at least one store type differs.
- Test: Kruskal-Wallis.
- groups = ['Express', 'Hypermarket', 'Supermarket']
- H = 6.874, p = 0.03216
- Decision: REJECT H0
- Business reading: store type is a meaningful segmentation for demand.

## Test 3 - Is stock-out associated with promotion status?
- H0: promotion and stock-out are independent.
- H1: they are associated.
- Test: Chi-square test of independence.
- chi2 = 0.000, p = 1, Cramer's V = nan
- Decision: FAIL TO REJECT H0
- Business reading: promotions do not measurably shift stock-out risk here.

## Test 4 - Does weekend demand differ from weekday demand?
- H0: same distribution weekday vs weekend.
- H1: distributions differ.
- Test: Mann-Whitney U.
- U = 544,375,352.0, p = 0.3002, Cohen's d = -0.005
- Decision: FAIL TO REJECT H0
- Business reading: weekend demand is not statistically different here.