"""Pydantic models with input validation for the autism prediction API."""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional


VALID_SEX_VALUES = {"m", "f"}
VALID_JAUNDICE_VALUES = {"yes", "no"}
VALID_FAMILY_ASD_VALUES = {"yes", "no"}
BINARY_FEATURE_KEYS = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"}
REQUIRED_FEATURES = {
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
    "Age_Mons", "Qchat-10-Score", "Sex", "Ethnicity",
    "Jaundice", "Family_mem_with_ASD", "Who completed the test",
}


class PredictRequest(BaseModel):
    """Request model for the /ml/predict endpoint."""

    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary with all required autism screening inputs.",
        examples=[{
            "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
            "A6": 0, "A7": 0, "A8": 0, "A9": 0, "A10": 0,
            "Age_Mons": 36, "Qchat-10-Score": 6,
            "Sex": "m", "Ethnicity": "White European",
            "Jaundice": "no", "Family_mem_with_ASD": "no",
            "Who completed the test": "family member",
        }],
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        missing = REQUIRED_FEATURES - set(v.keys())
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        # Validate binary answer columns (0 or 1)
        for key in BINARY_FEATURE_KEYS:
            if key in v and v[key] not in (0, 1):
                raise ValueError(f"Feature '{key}' must be 0 or 1, got {v[key]!r}")

        # Validate Sex field
        if "Sex" in v and str(v["Sex"]).lower() not in VALID_SEX_VALUES:
            raise ValueError(f"'Sex' must be one of {VALID_SEX_VALUES}, got {v['Sex']!r}")

        # Validate Jaundice field
        if "Jaundice" in v and str(v["Jaundice"]).lower() not in VALID_JAUNDICE_VALUES:
            raise ValueError(f"'Jaundice' must be one of {VALID_JAUNDICE_VALUES}, got {v['Jaundice']!r}")

        # Validate Family_mem_with_ASD field
        if "Family_mem_with_ASD" in v and str(v["Family_mem_with_ASD"]).lower() not in VALID_FAMILY_ASD_VALUES:
            raise ValueError(
                f"'Family_mem_with_ASD' must be one of {VALID_FAMILY_ASD_VALUES}, "
                f"got {v['Family_mem_with_ASD']!r}"
            )

        # Validate Age_Mons is a positive number
        if "Age_Mons" in v:
            try:
                age = float(v["Age_Mons"])
                if age < 0:
                    raise ValueError(f"'Age_Mons' must be a non-negative number, got {v['Age_Mons']!r}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"'Age_Mons' must be a number, got {v['Age_Mons']!r}") from exc

        return v


class PredictResponse(BaseModel):
    """Response model for the /ml/predict endpoint."""

    prediction: str = Field(..., description="Predicted class label ('Yes' or 'No').")
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence probability of the predicted class (0.0–1.0), if available.",
    )
