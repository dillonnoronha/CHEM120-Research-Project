"""
Regression tests for the security, correctness, and performance fixes in
pipeline.py: spreadsheet-formula neutralization on export, per-session chart
themes, descriptor edge cases, and batched / median-filled predictions.
"""

import io
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    add_chemical_descriptors,
    build_feature_matrix,
    chart_theme,
    clean_and_encode_data,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    email_member_match,
    load_atomic_table_from_bytes,
    load_en_table_from_bytes,
    load_radii_table_from_bytes,
    make_prediction_rows,
    make_single_prediction_row,
    neutralize_spreadsheet_formulas,
    set_chart_theme,
)


@pytest.fixture(scope="module")
def tables():
    return (load_atomic_table_from_bytes(None), load_en_table_from_bytes(None),
            load_radii_table_from_bytes(None))


# ---------------------------------------------------------------------------
# Export: CSV / formula injection
# ---------------------------------------------------------------------------

class TestFormulaNeutralization:

    def test_formula_text_gets_quoted(self):
        df = pd.DataFrame({"GroupNumber": ['=HYPERLINK("http://x","y")', "+1+1", "@SUM(A1)", "Group 4"]})
        out = neutralize_spreadsheet_formulas(df)
        assert out["GroupNumber"].tolist() == [
            '\'=HYPERLINK("http://x","y")', "'+1+1", "'@SUM(A1)", "Group 4"]

    def test_signed_numbers_and_numeric_columns_untouched(self):
        df = pd.DataFrame({"AN": ["-0.5", "+2", "1e-3"], "ON": [-1.0, 3.0, 4.0]})
        out = neutralize_spreadsheet_formulas(df)
        assert out["AN"].tolist() == ["-0.5", "+2", "1e-3"]
        assert out["ON"].tolist() == [-1.0, 3.0, 4.0]

    def test_input_frame_not_modified(self):
        df = pd.DataFrame({"Name": ["=1+1"]})
        neutralize_spreadsheet_formulas(df)
        assert df["Name"].iloc[0] == "=1+1"

    def test_csv_download_is_neutralized(self):
        text = dataframe_to_csv_bytes(pd.DataFrame({"Name": ["=cmd|' /C calc'!A0"]})).decode()
        assert "'=cmd" in text

    def test_excel_download_has_no_live_formula(self):
        data = dataframe_to_excel_bytes(pd.DataFrame({"Name": ["=1+1", "ok"]}))
        back = pd.read_excel(io.BytesIO(data))
        assert back["Name"].tolist() == ["'=1+1", "ok"]


# ---------------------------------------------------------------------------
# Chart theme is per session (per thread), not process-wide
# ---------------------------------------------------------------------------

def test_chart_theme_does_not_leak_between_threads():
    set_chart_theme("light")
    seen = {}
    worker = threading.Thread(target=lambda: seen.setdefault("other", chart_theme().ink))
    worker.start()
    worker.join()
    assert chart_theme().ink == "#24324a"          # this session: light
    assert seen["other"] == "#dbe4f3"              # another session: still the dark default
    set_chart_theme("dark")


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

class TestDescriptors:

    def test_mixed_site_flags_ignore_missing_values(self, tables):
        atomic, en, radii = tables
        df = pd.DataFrame({"A": ["La"], "AN": [2.0], "AP": [np.nan], "B": ["Ni"], "BN": [1.0],
                           "O": ["O"], "ON": [4.0]})
        out = add_chemical_descriptors(df, atomic, en, radii)
        assert out.loc[0, "Mixed_A_site"] == 0
        assert out.loc[0, "Mixed_B_site"] == 0
        assert out.loc[0, "Cation_count"] == 2

    def test_mixed_compound_values(self, tables):
        atomic, en, radii = tables
        row = make_single_prediction_row(atomic, en, "pure", "La", 1.0, "Sr", 1.0, "Fe", 1.0,
                                         "", 0.0, "", 0.0, 3.0, radii).iloc[0]
        la = atomic.set_index("Symbol").loc["La", "Z"]
        sr = atomic.set_index("Symbol").loc["Sr", "Z"]
        assert row["Formula"] == "LaSrFeO3"
        assert row["A_avg_Z"] == pytest.approx((la + sr) / 2)
        assert row["A_mix_fraction"] == pytest.approx(0.5)
        assert row["B_to_A_ratio"] == pytest.approx(0.5)

    def test_missing_required_ratio_gives_nan_not_crash(self, tables):
        atomic, en, radii = tables
        raw = pd.DataFrame([{"A": "La", "AN": "", "B": "Ni", "BN": 1, "O": "O", "ON": 4,
                             "P": "pure", "Bub": "yes"}])
        out = add_chemical_descriptors(clean_and_encode_data(raw, atomic), atomic, en, radii)
        assert np.isnan(out.loc[0, "A_total_ratio"])
        assert out.loc[0, "A_mix_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class TestPredictions:

    def test_batch_rows_match_single_rows(self, tables):
        atomic, en, radii = tables
        base = {"phase": "pure", "a": "La", "an": 2.0, "ap": "", "apn": 0.0, "b": "Ni", "bn": 1.0,
                "bp": "Co", "bpn": 0.5, "bdp": "", "bdpn": 0.0, "on": 4.0}
        values = [2.0, 3.5, 5.0]
        batch = make_prediction_rows(atomic, en, [{**base, "on": v} for v in values], radii)
        for i, v in enumerate(values):
            single = make_prediction_rows(atomic, en, [{**base, "on": v}], radii)
            pd.testing.assert_series_equal(batch.iloc[i], single.iloc[0], check_names=False)

    def test_fill_values_replace_blanks_before_zero_fill(self):
        row = pd.DataFrame({"Tolerance_factor": [np.nan], "AN": [np.nan]})
        plain = build_feature_matrix(row, use_phase=False, use_mass=False)
        filled = build_feature_matrix(row, use_phase=False, use_mass=False,
                                      fill_values={"Tolerance_factor": 0.95})
        assert plain.loc[0, "Tolerance_factor"] == 0
        assert filled.loc[0, "Tolerance_factor"] == 0.95
        assert filled.loc[0, "AN"] == 0


def test_email_member_match_is_stable_across_calls():
    first = email_member_match("jgarcia@csu.fullerton.edu", "Ana Lee, Jose Garcia")
    second = email_member_match("jgarcia@csu.fullerton.edu", "Ana Lee, Jose Garcia")
    assert first == second
    assert first[0] == "Jose Garcia"
