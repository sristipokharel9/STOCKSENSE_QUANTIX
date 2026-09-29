# StockSense - Data Quality Report

Every cleaning action is listed with its count, action and justification.

## Cleaning actions

| issue                                                           |   count_found | action                                                   | justification                                                              |
|:----------------------------------------------------------------|--------------:|:---------------------------------------------------------|:---------------------------------------------------------------------------|
| transactions: unparseable dates                                 |             0 | Rows dropped                                             | Cannot place them on the time axis; required for lag/rolling features.     |
| transactions: duplicate transaction_id (identical)              |          1002 | Dropped extra copies                                     | Same transaction recorded twice; safe to remove identical duplicates.      |
| transactions: negative quantity                                 |           299 | Excluded from demand (kept in rejected_rows.csv)         | Demand cannot be negative; treated as returns/data errors.                 |
| transactions: extreme quantity outliers (>99.9 pct per product) |           109 | Capped at 99.9th pct                                     | Extreme spikes break scaling for KNN/SVR; cap keeps signal, removes noise. |
| transactions: non-positive selling_price                        |             0 | Recomputed from median per product                       | Price must be positive; fallback to product median keeps revenue sane.     |
| transactions: rows after cleaning                               |         99992 | Removed 1301 rows                                        | Net effect of all cleaning steps on transactions.                          |
| products: category variants collapsed                           |             1 | Standardised to 5 categories                             | Consistent categories needed for grouping and feature engineering.         |
| inventory: unparseable dates                                    |             0 | Rows dropped                                             | Needed for time alignment with master table.                               |
| inventory: closing != opening + received - sold                 |          1193 | Reconciled closing = expected; flag kept                 | Arithmetic must hold for stock-out logic; flag retained for audit.         |
| inventory: store x product with sparse history                  |            15 | Flagged (threshold = 30 days)                            | Short history breaks lag features; flag allows low-confidence handling.    |
| external: missing temp_c                                        |           146 | Imputed with city-month median (fallback city -> global) | Weather values are slow-moving and correlated within city-month.           |
| external: missing rain_mm                                       |            73 | Imputed with city-month median (fallback city -> global) | Weather values are slow-moving and correlated within city-month.           |

## Injected trap log (Phase 0)

```json
{
  "missing_temp_c": 146,
  "missing_rain_mm": 73,
  "duplicate_transactions": 1002,
  "category_inconsistent": 1,
  "negative_quantity": 303,
  "extreme_quantity": 50,
  "inventory_mismatch": 1462,
  "sparse_products": [
    "P0008",
    "P0001",
    "P0014"
  ],
  "sparse_rows_removed": 10875,
  "messy_date_rows": 1012,
  "messy_store_id_rows": 1012,
  "seed": 42
}
```

## Notes

- Rejected rows are stored in `data/processed/rejected_rows.csv`.
- `inventory_mismatch` is kept as an audit feature.
- Sparse-history store x product combos are flagged for low-confidence predictions.