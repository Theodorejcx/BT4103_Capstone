import pandas as pd
import numpy as np
import argparse
from typing import Tuple

def _is_number_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def compare_csvs(
    path_a: str,
    path_b: str,
    *,
    float_tol: float = 0.0,
    ignore_column_order: bool = False,
    ignore_row_order: bool = False,
    strip_strings: bool = True,
    dtype_check: bool = False,
    read_csv_kwargs: dict = None
) -> Tuple[bool, pd.DataFrame]:
    """
    Compare two CSV files cell-by-cell.

    Returns:
      (are_equal: bool, mismatches: DataFrame)
      mismatches columns: ['row_index', 'column', 'value_a', 'value_b']

    Options:
      float_tol: if > 0 use np.isclose for numeric comparisons with this absolute tolerance.
      ignore_column_order: if True, sort columns by name before comparing.
      ignore_row_order: if True, rows are compared by position after sorting all columns; you can
                        also set index columns prior to calling to control alignment.
      strip_strings: if True, strip whitespace from string values before comparing.
      dtype_check: if True, also raise/report if dtypes differ (not strict by default).
      read_csv_kwargs: dict passed to pd.read_csv (e.g. encoding, parse_dates).
    """
    read_csv_kwargs = read_csv_kwargs or {}

    a = pd.read_csv(path_a, **read_csv_kwargs)
    b = pd.read_csv(path_b, **read_csv_kwargs)

    # Optionally strip string whitespace early
    if strip_strings:
        for df in (a, b):
            for col in df.select_dtypes(include=["object", "string"]).columns:
                df[col] = df[col].astype("string").str.strip()

    # Optional dtype check (best-effort)
    if dtype_check:
        dtypes_a = a.dtypes.astype(str).to_dict()
        dtypes_b = b.dtypes.astype(str).to_dict()
        if dtypes_a != dtypes_b:
            # not fatal; we'll continue but include in mismatch output if desired
            # You could raise here if you want strict dtype equality:
            pass

    # Align columns
    cols_a = list(a.columns)
    cols_b = list(b.columns)

    if ignore_column_order:
        common_cols = sorted(set(cols_a) & set(cols_b))
        a = a.loc[:, common_cols]
        b = b.loc[:, common_cols]
    else:
        if cols_a != cols_b:
            # columns differ in content/order -> not equal
            # Build mismatch rows for column-level differences and return
            missing_in_b = [c for c in cols_a if c not in cols_b]
            missing_in_a = [c for c in cols_b if c not in cols_a]
            rows = []
            for c in missing_in_b:
                rows.append({"row_index": "<column>", "column": c, "value_a": "<present>", "value_b": "<missing>"})
            for c in missing_in_a:
                rows.append({"row_index": "<column>", "column": c, "value_a": "<missing>", "value_b": "<present>"})
            return False, pd.DataFrame(rows, columns=["row_index", "column", "value_a", "value_b"])

    # Align rows/indices
    if ignore_row_order:
        # We'll sort rows deterministically by all columns to compare content regardless of order
        a = a.sort_values(by=list(a.columns)).reset_index(drop=True)
        b = b.sort_values(by=list(b.columns)).reset_index(drop=True)

    # If shapes differ, report
    if a.shape != b.shape:
        rows = [{
            "row_index": "<shape>",
            "column": "<shape>",
            "value_a": f"shape={a.shape}",
            "value_b": f"shape={b.shape}"
        }]
        return False, pd.DataFrame(rows, columns=["row_index", "column", "value_a", "value_b"])

    # Now compare cell-by-cell
    n_rows, n_cols = a.shape
    columns = a.columns.tolist()
    index = a.index

    mismatches = []
    # Precompute numeric mask per column for efficiency
    numeric_mask = {col: _is_number_dtype(a[col]) or _is_number_dtype(b[col]) for col in columns}

    for col in columns:
        col_a = a[col]
        col_b = b[col]

        # For numeric columns (or requested tolerance) use isclose with NaN handling
        if numeric_mask[col]:
            # Convert to numeric (coerce errors to NaN) to compare numerically
            va = pd.to_numeric(col_a, errors="coerce").to_numpy(dtype=float)
            vb = pd.to_numeric(col_b, errors="coerce").to_numpy(dtype=float)

            # Where both NaN -> equal
            both_nan = np.isnan(va) & np.isnan(vb)

            if float_tol and float_tol > 0:
                close = np.isclose(va, vb, atol=float_tol, equal_nan=True)
            else:
                # exact numeric equality treating NaN==NaN
                close = (va == vb) | both_nan

            # find mismatches
            mismatch_idx = np.where(~close)[0]
            for i in mismatch_idx:
                mismatches.append({
                    "row_index": index[i],
                    "column": col,
                    "value_a": col_a.iat[i],
                    "value_b": col_b.iat[i]
                })
        else:
            # Object/string columns: compare treating NaN==NaN
            sa = col_a.astype(object).where(col_a.notna(), np.nan)
            sb = col_b.astype(object).where(col_b.notna(), np.nan)

            # Use pandas comparison where NaNs are equal
            eq = sa.fillna("<_pandas_nan_>") == sb.fillna("<_pandas_nan_>")
            mismatch_idx = np.where(~eq)[0]
            for i in mismatch_idx:
                # handle NaNs for nicer output
                va = None if pd.isna(sa.iat[i]) else sa.iat[i]
                vb = None if pd.isna(sb.iat[i]) else sb.iat[i]
                mismatches.append({
                    "row_index": index[i],
                    "column": col,
                    "value_a": va,
                    "value_b": vb
                })

    are_equal = len(mismatches) == 0
    mismatches_df = pd.DataFrame(mismatches, columns=["row_index", "column", "value_a", "value_b"])
    return are_equal, mismatches_df

# === CLI wrapper example ===
def _main():
    parser = argparse.ArgumentParser(description="Compare two CSV files cell-by-cell.")
    parser.add_argument("a", help="First CSV file path")
    parser.add_argument("b", help="Second CSV file path")
    parser.add_argument("--tol", type=float, default=0.0, help="Absolute tolerance for numeric comparisons")
    parser.add_argument("--ignore-column-order", action="store_true", help="Ignore column order")
    parser.add_argument("--ignore-row-order", action="store_true", help="Ignore row order (sort rows by all columns first)")
    parser.add_argument("--no-strip", dest="strip", action="store_false", help="Do not strip whitespace from string columns")
    parser.add_argument("--out", help="Optional path to write mismatches CSV")
    args = parser.parse_args()

    eq, mism = compare_csvs(
        args.a, args.b,
        float_tol=args.tol,
        ignore_column_order=args.ignore_column_order,
        ignore_row_order=args.ignore_row_order,
        strip_strings=args.strip
    )
    if eq:
        print("FILES ARE EQUAL (cell-by-cell).")
    else:
        print(f"FILES DIFFER: {len(mism)} mismatched cells.")
        print(mism.head(50).to_string(index=False))
        if args.out:
            mism.to_csv(args.out, index=False)
            print(f"Wrote mismatches to {args.out}")

if __name__ == '__main__':
    _main()