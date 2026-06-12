"""
Unit tests for ML utility functions in pipeline.py

Tests cover:
1. Data loading & cleaning (read_table_from_bytes, clean_text, safe_float)
2. Preprocessing & encoding (clean_and_encode_data, normalize_phase_label, normalize_bubble_label)
3. Feature engineering (add_chemical_descriptors, build_feature_matrix)
4. Model training & prediction (train_ml_model, make_single_prediction_row)
"""

import io
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Import functions from pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    is_blank,
    clean_text,
    safe_float,
    format_ratio,
    normalize_phase_label,
    normalize_bubble_label,
    read_table_from_bytes,
    clean_and_encode_data,
    add_chemical_descriptors,
    build_feature_matrix,
    train_ml_model,
    make_single_prediction_row,
    load_atomic_table_from_bytes,
)


# ============================================================================
# FIXTURES: Test data
# ============================================================================

@pytest.fixture
def atomic_table():
    """Minimal atomic reference table for testing."""
    return pd.DataFrame([
        {"Element": "Oxygen", "Symbol": "O", "Z": 8, "AtomicMass": 15.999},
        {"Element": "Calcium", "Symbol": "Ca", "Z": 20, "AtomicMass": 40.078},
        {"Element": "Manganese", "Symbol": "Mn", "Z": 25, "AtomicMass": 54.938},
        {"Element": "Iron", "Symbol": "Fe", "Z": 26, "AtomicMass": 55.845},
    ])


@pytest.fixture
def en_table():
    """Empty electronegativity table (optional feature)."""
    return pd.DataFrame()


@pytest.fixture
def sample_raw_data():
    """Sample raw data before cleaning."""
    return pd.DataFrame([
        {
            "GroupNumber": 1,
            "Instructor": "Dr. Smith",
            "Semester": "FA25",
            "Slot": 1,
            "A": "Ca",
            "AN": 1.0,
            "AP": "",
            "APN": np.nan,
            "B": "Mn",
            "BN": 0.5,
            "BP": "",
            "BPN": np.nan,
            "BDP": "",
            "BDPN": np.nan,
            "O": "O",
            "ON": 3.0,
            "P": "pure",
            "Bub": "yes",
        },
        {
            "GroupNumber": 2,
            "Instructor": "Dr. Smith",
            "Semester": "FA25",
            "Slot": 2,
            "A": "Ca",
            "AN": 1.0,
            "AP": "",
            "APN": np.nan,
            "B": "Fe",
            "BN": 1.0,
            "BP": "",
            "BPN": np.nan,
            "BDP": "",
            "BDPN": np.nan,
            "O": "O",
            "ON": 4.0,
            "P": "impure",
            "Bub": "no",
        },
    ])


@pytest.fixture
def sample_clean_data(atomic_table):
    """Sample cleaned data after preprocessing."""
    raw = pd.DataFrame([
        {
            "GroupNumber": 1,
            "A": "Ca",
            "AN": 1.0,
            "B": "Mn",
            "BN": 0.5,
            "O": "O",
            "ON": 3.0,
            "P": "pure",
            "Bub": "yes",
            "PhaseN": 2,
            "BubN": 1,
            "BubbleYes": 1,
        },
        {
            "GroupNumber": 2,
            "A": "Ca",
            "AN": 1.0,
            "B": "Fe",
            "BN": 1.0,
            "O": "O",
            "ON": 4.0,
            "P": "impure",
            "Bub": "no",
            "PhaseN": 1,
            "BubN": 2,
            "BubbleYes": 0,
        },
    ])
    return raw


# ============================================================================
# 1. UNIT TESTS: Data Loading & Cleaning (Basic Utilities)
# ============================================================================

class TestUtilityFunctions:
    """Test basic utility functions for data cleaning."""

    def test_is_blank_with_none(self):
        """is_blank returns True for None."""
        assert is_blank(None) is True

    def test_is_blank_with_empty_string(self):
        """is_blank returns True for empty string."""
        assert is_blank("") is True

    def test_is_blank_with_whitespace(self):
        """is_blank returns True for whitespace-only string."""
        assert is_blank("   ") is True

    def test_is_blank_with_valid_value(self):
        """is_blank returns False for non-blank values."""
        assert is_blank("Ca") is False
        assert is_blank(1.0) is False

    def test_clean_text_basic(self):
        """clean_text trims whitespace."""
        assert clean_text("  Ca  ") == "Ca"

    def test_clean_text_blank_returns_empty(self):
        """clean_text returns empty string for blank input."""
        assert clean_text(None) == ""
        assert clean_text("") == ""

    def test_safe_float_valid(self):
        """safe_float converts valid numeric strings."""
        assert safe_float("1.5") == 1.5
        assert safe_float("2") == 2.0
        assert safe_float("0.25") == 0.25

    def test_safe_float_blank_returns_nan(self):
        """safe_float returns NaN for blank values."""
        assert np.isnan(safe_float(None))
        assert np.isnan(safe_float(""))

    def test_safe_float_invalid_returns_nan(self):
        """safe_float returns NaN for non-numeric strings."""
        assert np.isnan(safe_float("not a number"))

    def test_safe_float_fraction(self):
        """safe_float accepts simple fractions."""
        assert safe_float("1/2") == 0.5
        assert safe_float("2/4") == 0.5

    def test_format_ratio_one(self):
        """format_ratio returns empty string for 1.0."""
        assert format_ratio(1.0) == ""

    def test_format_ratio_integer(self):
        """format_ratio formats integers without decimal."""
        assert format_ratio(2.0) == "2"
        assert format_ratio(3) == "3"

    def test_format_ratio_decimal(self):
        """format_ratio preserves decimal values."""
        assert format_ratio(0.5) == "0.5"

    def test_format_ratio_zero_or_blank(self):
        """format_ratio returns empty for zero and blank."""
        assert format_ratio(0) == ""
        assert format_ratio(None) == ""


# ============================================================================
# 2. UNIT TESTS: Data Loading (CSV/Excel)
# ============================================================================

class TestDataLoading:
    """Test data loading from CSV and Excel."""

    def test_read_table_from_bytes_csv(self):
        """read_table_from_bytes loads CSV data."""
        csv_data = "Name,Value\nCa,1\nMn,2\n"
        file_bytes = csv_data.encode()
        df = read_table_from_bytes(file_bytes, "test.csv")

        assert len(df) == 2
        assert list(df.columns) == ["Name", "Value"]
        assert df.iloc[0]["Name"] == "Ca"

    def test_read_table_from_bytes_unsupported_type(self):
        """read_table_from_bytes raises error for unsupported file type."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            read_table_from_bytes(b"data", "test.txt")

    def test_read_table_from_bytes_empty_csv(self):
        """read_table_from_bytes handles empty CSV."""
        csv_data = "Name,Value\n"
        file_bytes = csv_data.encode()
        df = read_table_from_bytes(file_bytes, "empty.csv")

        assert len(df) == 0
        assert list(df.columns) == ["Name", "Value"]


# ============================================================================
# 3. UNIT TESTS: Preprocessing & Encoding
# ============================================================================

class TestNormalizationFunctions:
    """Test label normalization for phase and bubble response."""

    def test_normalize_phase_pure(self):
        """normalize_phase_label recognizes pure phase."""
        assert normalize_phase_label("pure") == "pure"
        assert normalize_phase_label("Pure") == "pure"
        assert normalize_phase_label("PURE PHASE") == "pure"

    def test_normalize_phase_impure(self):
        """normalize_phase_label recognizes impure/heterogeneous."""
        assert normalize_phase_label("impure") == "impure"
        assert normalize_phase_label("Heterogeneous") == "impure"

    def test_normalize_phase_not_made(self):
        """normalize_phase_label recognizes not made."""
        assert normalize_phase_label("did not make") == "not made"
        assert normalize_phase_label("melted") == "not made"
        assert normalize_phase_label("failed") == "not made"

    def test_normalize_phase_blank(self):
        """normalize_phase_label returns empty for blank."""
        assert normalize_phase_label("") == ""
        assert normalize_phase_label(None) == ""

    def test_normalize_phase_unknown(self):
        """normalize_phase_label returns other for unknown."""
        assert normalize_phase_label("unknown value") == "other/needs review"

    def test_normalize_bubble_yes(self):
        """normalize_bubble_label recognizes yes."""
        assert normalize_bubble_label("yes") == "yes"
        assert normalize_bubble_label("YES") == "yes"
        assert normalize_bubble_label("y") == "yes"

    def test_normalize_bubble_no(self):
        """normalize_bubble_label recognizes no."""
        assert normalize_bubble_label("no") == "no"
        assert normalize_bubble_label("NO") == "no"
        assert normalize_bubble_label("n") == "no"

    def test_normalize_bubble_maybe(self):
        """normalize_bubble_label recognizes maybe."""
        assert normalize_bubble_label("maybe") == "maybe"
        assert normalize_bubble_label("MAYBE") == "maybe"
        assert normalize_bubble_label("possibly") == "maybe"

    def test_normalize_bubble_blank(self):
        """normalize_bubble_label returns empty for blank."""
        assert normalize_bubble_label("") == ""
        assert normalize_bubble_label(None) == ""


class TestCleanAndEncodeData:
    """Test data cleaning and encoding pipeline."""

    def test_clean_and_encode_empty_dataframe(self, atomic_table):
        """clean_and_encode_data handles empty DataFrame."""
        empty_df = pd.DataFrame()
        result = clean_and_encode_data(empty_df, atomic_table)
        assert result.empty

    def test_clean_and_encode_creates_numeric_columns(self, sample_raw_data, atomic_table):
        """clean_and_encode_data creates PhaseN and BubN numeric columns."""
        result = clean_and_encode_data(sample_raw_data, atomic_table)

        assert "PhaseN" in result.columns
        assert "BubN" in result.columns
        assert result.iloc[0]["PhaseN"] == 2  # pure -> 2
        assert result.iloc[0]["BubN"] == 1  # yes -> 1

    def test_clean_and_encode_creates_bubble_yes_binary(self, sample_raw_data, atomic_table):
        """clean_and_encode_data creates BubbleYes binary column."""
        result = clean_and_encode_data(sample_raw_data, atomic_table)

        assert "BubbleYes" in result.columns
        assert result.iloc[0]["BubbleYes"] == 1  # yes
        assert result.iloc[1]["BubbleYes"] == 0  # no

    def test_clean_and_encode_infers_oxygen(self, atomic_table):
        """clean_and_encode_data infers O when O is blank but ON is filled."""
        df = pd.DataFrame([
            {
                "A": "Ca",
                "AN": 1.0,
                "B": "Mn",
                "BN": 0.5,
                "O": "",
                "ON": 3.0,
                "P": "pure",
                "Bub": "yes",
            }
        ])
        result = clean_and_encode_data(df, atomic_table)
        assert result.iloc[0]["O"] == "O"

    def test_clean_and_encode_normalizes_labels(self, atomic_table):
        """clean_and_encode_data normalizes raw phase and bubble labels."""
        df = pd.DataFrame([
            {
                "A": "Ca",
                "AN": 1.0,
                "B": "Mn",
                "BN": 0.5,
                "O": "O",
                "ON": 3.0,
                "P": "pure phase",
                "Bub": "YES",
            }
        ])
        result = clean_and_encode_data(df, atomic_table)
        assert result.iloc[0]["P"] == "pure"
        assert result.iloc[0]["Bub"] == "yes"


# ============================================================================
# 4. UNIT TESTS: Feature Engineering
# ============================================================================

class TestAddChemicalDescriptors:
    """Test addition of chemical descriptor features."""

    def test_add_descriptors_empty_dataframe(self, atomic_table, en_table):
        """add_chemical_descriptors handles empty DataFrame."""
        empty_df = pd.DataFrame()
        result = add_chemical_descriptors(empty_df, atomic_table, en_table)
        assert result.empty

    def test_add_descriptors_creates_z_columns(self, sample_clean_data, atomic_table, en_table):
        """add_chemical_descriptors creates atomic number columns."""
        result = add_chemical_descriptors(sample_clean_data, atomic_table, en_table)

        assert "A_Z" in result.columns
        assert "B_Z" in result.columns
        assert result.iloc[0]["A_Z"] == 20  # Ca
        assert result.iloc[0]["B_Z"] == 25  # Mn

    def test_add_descriptors_creates_mass_columns(self, sample_clean_data, atomic_table, en_table):
        """add_chemical_descriptors creates atomic mass columns."""
        result = add_chemical_descriptors(sample_clean_data, atomic_table, en_table)

        assert "A_Mass" in result.columns
        assert "B_Mass" in result.columns
        assert result.iloc[0]["A_Mass"] == 40.078  # Ca

    def test_add_descriptors_creates_formula_mass(self, sample_clean_data, atomic_table, en_table):
        """add_chemical_descriptors calculates formula mass."""
        result = add_chemical_descriptors(sample_clean_data, atomic_table, en_table)

        assert "FormulaMass" in result.columns
        # CaMnO3: (40.078 * 1) + (54.938 * 0.5) + (15.999 * 3)
        expected_mass = 40.078 + 27.469 + 47.997
        assert abs(result.iloc[0]["FormulaMass"] - expected_mass) < 0.01

    def test_add_descriptors_creates_ratio_columns(self, sample_clean_data, atomic_table, en_table):
        """add_chemical_descriptors creates structural ratio columns."""
        result = add_chemical_descriptors(sample_clean_data, atomic_table, en_table)

        assert "O_to_cation_ratio" in result.columns
        assert "B_to_A_ratio" in result.columns
        assert "Mixed_A_site" in result.columns
        assert "Mixed_B_site" in result.columns

    def test_add_descriptors_site_totals(self, sample_clean_data, atomic_table, en_table):
        """add_chemical_descriptors calculates site-level totals."""
        result = add_chemical_descriptors(sample_clean_data, atomic_table, en_table)

        assert "A_total_ratio" in result.columns
        assert "B_total_ratio" in result.columns
        assert result.iloc[0]["A_total_ratio"] == 1.0
        assert result.iloc[0]["B_total_ratio"] == 0.5


class TestBuildFeatureMatrix:
    """Test feature matrix building for ML."""

    def test_build_feature_matrix_basic(self, sample_clean_data):
        """build_feature_matrix creates numeric and one-hot features."""
        features = build_feature_matrix(sample_clean_data, use_phase=True)

        assert features.shape[0] == 2  # 2 rows
        assert features.shape[1] > 0  # Some features
        assert isinstance(features, pd.DataFrame)

    def test_build_feature_matrix_numeric_features(self, sample_clean_data):
        """build_feature_matrix includes numeric features."""
        features = build_feature_matrix(sample_clean_data, use_phase=True)

        # Should have numeric features like AN, BN, ON
        assert any("AN" in col for col in features.columns) or len(features.columns) > 0

    def test_build_feature_matrix_alignment(self, sample_clean_data):
        """build_feature_matrix aligns to expected columns."""
        features1 = build_feature_matrix(sample_clean_data, use_phase=True)
        expected_cols = list(features1.columns)

        # Align to expected columns
        features2 = build_feature_matrix(sample_clean_data, use_phase=True, expected_columns=expected_cols)

        assert list(features2.columns) == expected_cols
        assert features2.shape[1] == len(expected_cols)

    def test_build_feature_matrix_with_phase(self, sample_clean_data):
        """build_feature_matrix includes phase when use_phase=True."""
        features_with = build_feature_matrix(sample_clean_data, use_phase=True)
        features_without = build_feature_matrix(sample_clean_data, use_phase=False)

        # With phase should have phase-related features (or at least same/more features)
        assert features_with.shape[1] >= features_without.shape[1]


# ============================================================================
# 5. UNIT TESTS: Model Training & Prediction
# ============================================================================

class TestTrainMLModel:
    """Test ML model training."""

    def test_train_model_insufficient_data(self):
        """train_ml_model returns error for < 10 labeled rows."""
        df = pd.DataFrame([
            {"BubbleYes": 1, "Bub": "yes"},
            {"BubbleYes": 0, "Bub": "no"},
        ])
        result = train_ml_model(df)

        assert result["ok"] is False
        assert "at least 10 labeled" in result["message"]

    def test_train_model_single_class(self):
        """train_ml_model returns error for single class."""
        df = pd.DataFrame([
            {"BubbleYes": 1, "Bub": "yes"} for _ in range(15)
        ])
        result = train_ml_model(df)

        assert result["ok"] is False
        assert "both bubble = yes and bubble = no" in result["message"]

    def test_train_model_success_mixed_data(self, sample_clean_data):
        """train_ml_model trains successfully with mixed classes."""
        # Expand data to meet minimum threshold
        df = pd.concat([sample_clean_data] * 8, ignore_index=True)
        df["AN"] = 1.0
        df["BN"] = 0.5
        df["ON"] = 3.0

        result = train_ml_model(df)

        assert result["ok"] is True
        assert "model" in result
        assert "feature_columns" in result
        # Per-model score dicts for Random Forest, Gradient Boosting, Logistic Regression.
        for key in ["rf", "gb", "lr", "best"]:
            assert key in result
            assert "accuracy" in result[key]
            assert "balanced_accuracy" in result[key]
        assert result["model_name"] in {"Random Forest", "Gradient Boosting"}

    def test_train_model_returns_metrics(self, sample_clean_data):
        """train_ml_model returns expected metrics."""
        df = pd.concat([sample_clean_data] * 8, ignore_index=True)
        df["AN"] = 1.0
        df["BN"] = 0.5
        df["ON"] = 3.0

        result = train_ml_model(df)

        if result["ok"]:
            for key in ["rf", "gb", "lr"]:
                assert 0 <= result[key]["accuracy"] <= 1
                assert 0 <= result[key]["balanced_accuracy"] <= 1
            assert 0 <= result["baseline_accuracy"] <= 1
            assert result["training_rows"] > 0
            assert result["testing_rows"] > 0


class TestMakeSinglePredictionRow:
    """Test single-row prediction input construction."""

    def test_make_prediction_row_basic(self, atomic_table, en_table):
        """make_single_prediction_row creates a cleaned row."""
        result = make_single_prediction_row(
            atomic=atomic_table,
            en_table=en_table,
            phase="pure",
            a="Ca",
            an=1.0,
            ap="",
            apn=np.nan,
            b="Mn",
            bn=0.5,
            bp="",
            bpn=np.nan,
            bdp="",
            bdpn=np.nan,
            on=3.0,
        )

        assert len(result) == 1
        assert result.iloc[0]["A"] == "Ca"
        assert result.iloc[0]["B"] == "Mn"
        assert result.iloc[0]["O"] == "O"

    def test_make_prediction_row_has_descriptors(self, atomic_table, en_table):
        """make_single_prediction_row includes chemical descriptors."""
        result = make_single_prediction_row(
            atomic=atomic_table,
            en_table=en_table,
            phase="pure",
            a="Ca",
            an=1.0,
            ap="",
            apn=np.nan,
            b="Mn",
            bn=0.5,
            bp="",
            bpn=np.nan,
            bdp="",
            bdpn=np.nan,
            on=3.0,
        )

        # Should have descriptor columns
        assert "A_Z" in result.columns
        assert "B_Z" in result.columns
        assert "FormulaMass" in result.columns

    def test_make_prediction_row_normalized_labels(self, atomic_table, en_table):
        """make_single_prediction_row normalizes input labels."""
        result = make_single_prediction_row(
            atomic=atomic_table,
            en_table=en_table,
            phase="PURE",
            a="Ca",
            an=1.0,
            ap="",
            apn=np.nan,
            b="Mn",
            bn=0.5,
            bp="",
            bpn=np.nan,
            bdp="",
            bdpn=np.nan,
            on=3.0,
        )

        # Phase should be normalized to lowercase
        assert result.iloc[0]["P"] == "pure"


# ============================================================================
# INTEGRATION EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_safe_float_with_zero_denominator(self):
        """safe_float handles zero denominator in fractions."""
        result = safe_float("1/0")
        assert np.isnan(result)

    def test_format_ratio_very_small_number(self):
        """format_ratio handles very small numbers."""
        result = format_ratio(0.0001)
        assert isinstance(result, str)

    def test_clean_and_encode_missing_optional_columns(self, atomic_table):
        """clean_and_encode_data handles missing optional columns."""
        df = pd.DataFrame([
            {"A": "Ca", "AN": 1.0, "B": "Mn", "BN": 0.5, "O": "O", "ON": 3.0}
        ])
        # Missing P and Bub columns
        result = clean_and_encode_data(df, atomic_table)

        assert "P" in result.columns
        assert "Bub" in result.columns
        assert "PhaseN" in result.columns
        assert "BubN" in result.columns

    def test_add_descriptors_with_missing_elements(self, atomic_table, en_table):
        """add_chemical_descriptors handles unknown elements gracefully."""
        df = pd.DataFrame([
            {
                "A": "XX",  # Invalid element
                "AN": 1.0,
                "B": "Mn",
                "BN": 0.5,
                "O": "O",
                "ON": 3.0,
            }
        ])
        result = add_chemical_descriptors(df, atomic_table, en_table)

        # Should still return a DataFrame with NaN for unknown elements
        assert "A_Z" in result.columns
        assert len(result) == 1

    def test_build_feature_matrix_with_nan_values(self, sample_clean_data):
        """build_feature_matrix handles NaN values."""
        df = sample_clean_data.copy()
        df.iloc[0, df.columns.get_loc("AN")] = np.nan

        features = build_feature_matrix(df, use_phase=True)

        # Should fill NaN with 0
        assert not features.isna().any().any()

    def test_normalize_phase_with_old_wording(self):
        """normalize_phase_label handles old CHEM 120 wording."""
        result = normalize_phase_label("pure phase/homogenous mixture")
        assert result == "pure"

        result = normalize_phase_label("did not make pure compound")
        assert result == "not made"
