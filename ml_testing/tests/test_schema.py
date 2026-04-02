"""Unit tests for Pydantic request/response schema validation."""

import pytest
from pydantic import ValidationError

from app.schema import PredictRequest, PredictResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_features() -> dict:
    """Return a complete, valid feature dictionary."""
    return {
        "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
        "A6": 0, "A7": 0, "A8": 0, "A9": 0, "A10": 0,
        "Age_Mons": 36,
        "Qchat-10-Score": 6,
        "Sex": "m",
        "Ethnicity": "White European",
        "Jaundice": "no",
        "Family_mem_with_ASD": "no",
        "Who completed the test": "family member",
    }


# ---------------------------------------------------------------------------
# PredictRequest – valid inputs
# ---------------------------------------------------------------------------

class TestPredictRequestValid:
    def test_all_zeros(self):
        req = PredictRequest(features=_valid_features())
        assert req.features["A1"] == 0

    def test_all_ones(self):
        feat = _valid_features()
        for i in range(1, 11):
            feat[f"A{i}"] = 1
        feat["Qchat-10-Score"] = 10
        req = PredictRequest(features=feat)
        assert req.features["A10"] == 1

    def test_sex_female(self):
        feat = _valid_features()
        feat["Sex"] = "f"
        req = PredictRequest(features=feat)
        assert req.features["Sex"] == "f"

    def test_jaundice_yes(self):
        feat = _valid_features()
        feat["Jaundice"] = "yes"
        req = PredictRequest(features=feat)
        assert req.features["Jaundice"] == "yes"

    def test_family_yes(self):
        feat = _valid_features()
        feat["Family_mem_with_ASD"] = "yes"
        req = PredictRequest(features=feat)
        assert req.features["Family_mem_with_ASD"] == "yes"

    def test_age_zero(self):
        feat = _valid_features()
        feat["Age_Mons"] = 0
        req = PredictRequest(features=feat)
        assert req.features["Age_Mons"] == 0

    def test_age_max(self):
        feat = _valid_features()
        feat["Age_Mons"] = 300
        req = PredictRequest(features=feat)
        assert req.features["Age_Mons"] == 300

    def test_who_healthcare(self):
        feat = _valid_features()
        feat["Who completed the test"] = "healthcare professional"
        req = PredictRequest(features=feat)
        assert req.features["Who completed the test"] == "healthcare professional"

    def test_who_self(self):
        feat = _valid_features()
        feat["Who completed the test"] = "self"
        req = PredictRequest(features=feat)
        assert req.features["Who completed the test"] == "self"

    def test_who_others(self):
        feat = _valid_features()
        feat["Who completed the test"] = "others"
        req = PredictRequest(features=feat)
        assert req.features["Who completed the test"] == "others"


# ---------------------------------------------------------------------------
# PredictRequest – invalid inputs trigger ValidationError
# ---------------------------------------------------------------------------

class TestPredictRequestInvalid:
    def test_a_value_out_of_range(self):
        feat = _valid_features()
        feat["A3"] = 5  # must be 0 or 1
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_a_value_negative(self):
        feat = _valid_features()
        feat["A1"] = -1
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_age_negative(self):
        feat = _valid_features()
        feat["Age_Mons"] = -5
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_age_over_max(self):
        feat = _valid_features()
        feat["Age_Mons"] = 400
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_qchat_over_max(self):
        feat = _valid_features()
        feat["Qchat-10-Score"] = 11
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_qchat_negative(self):
        feat = _valid_features()
        feat["Qchat-10-Score"] = -1
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_invalid_sex(self):
        feat = _valid_features()
        feat["Sex"] = "x"
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_invalid_jaundice(self):
        feat = _valid_features()
        feat["Jaundice"] = "maybe"
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_invalid_family(self):
        feat = _valid_features()
        feat["Family_mem_with_ASD"] = "unknown"
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_invalid_who(self):
        feat = _valid_features()
        feat["Who completed the test"] = "robot"
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_non_numeric_age(self):
        feat = _valid_features()
        feat["Age_Mons"] = "old"
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)

    def test_non_numeric_qchat(self):
        feat = _valid_features()
        feat["Qchat-10-Score"] = "ten"
        with pytest.raises(ValidationError):
            PredictRequest(features=feat)


# ---------------------------------------------------------------------------
# PredictResponse
# ---------------------------------------------------------------------------

class TestPredictResponse:
    def test_valid_yes(self):
        resp = PredictResponse(prediction="Yes", probability=0.92)
        assert resp.prediction == "Yes"
        assert resp.probability == pytest.approx(0.92)

    def test_valid_no_probability(self):
        resp = PredictResponse(prediction="No", probability=None)
        assert resp.prediction == "No"
        assert resp.probability is None

    def test_probability_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictResponse(prediction="Yes", probability=1.5)

    def test_probability_negative(self):
        with pytest.raises(ValidationError):
            PredictResponse(prediction="Yes", probability=-0.1)
