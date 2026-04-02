"""Pydantic request/response models for the Autism Prediction API.

All user-supplied data is validated here before reaching the service layer.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Example payload used in the interactive API docs (Swagger UI)
# ---------------------------------------------------------------------------
_EXAMPLE_FEATURES: dict[str, Any] = {
    "A1": 0,
    "A2": 0,
    "A3": 0,
    "A4": 0,
    "A5": 0,
    "A6": 0,
    "A7": 0,
    "A8": 0,
    "A9": 0,
    "A10": 0,
    "Age_Mons": 36,
    "Qchat-10-Score": 6,
    "Sex": "m",
    "Ethnicity": "White European",
    "Jaundice": "no",
    "Family_mem_with_ASD": "no",
    "Who completed the test": "family member",
}

# Allowed values for categorical fields
_VALID_SEX = {"m", "f"}
_VALID_JAUNDICE = {"yes", "no"}
_VALID_FAMILY = {"yes", "no"}
_VALID_ETHNICITY = {
    "white european",
    "asian",
    "black",
    "latino",
    "middle eastern",
    "mixed",
    "others",
    "south asian",
    "hispanic",
    "native indian",
    "pacifica",
}
_VALID_WHO = {
    "family member",
    "healthcare professional",
    "self",
    "others",
}


class PredictRequest(BaseModel):
    """Request body for the ``POST /ml/predict`` endpoint.

    Attributes:
        features: A flat dictionary mapping feature names to their values.
            Numeric screening questions ``A1``–``A10`` must be ``0`` or ``1``.
            ``Age_Mons`` must be a non-negative integer.
            ``Qchat-10-Score`` must be between ``0`` and ``10``.
    """

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Feature dictionary containing all required input fields. "
            "See /ml/metadata for the full list of expected features."
        ),
        json_schema_extra={"example": _EXAMPLE_FEATURES},
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate individual feature values."""
        errors: list[str] = []

        # Validate binary screening questions A1..A10
        for i in range(1, 11):
            key = f"A{i}"
            if key in v:
                try:
                    val = int(v[key])
                    if val not in (0, 1):
                        errors.append(f"{key} must be 0 or 1, got {val!r}")
                except (TypeError, ValueError):
                    errors.append(f"{key} must be an integer (0 or 1), got {v[key]!r}")

        # Validate age
        if "Age_Mons" in v:
            try:
                age = int(v["Age_Mons"])
                if age < 0 or age > 300:
                    errors.append(
                        f"Age_Mons must be between 0 and 300 months, got {age}"
                    )
            except (TypeError, ValueError):
                errors.append(f"Age_Mons must be numeric, got {v['Age_Mons']!r}")

        # Validate Qchat score
        if "Qchat-10-Score" in v:
            try:
                score = int(v["Qchat-10-Score"])
                if score < 0 or score > 10:
                    errors.append(
                        f"Qchat-10-Score must be between 0 and 10, got {score}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Qchat-10-Score must be numeric, got {v['Qchat-10-Score']!r}"
                )

        # Validate categorical fields (case-insensitive)
        if "Sex" in v and str(v["Sex"]).lower() not in _VALID_SEX:
            errors.append(f"Sex must be one of {sorted(_VALID_SEX)}, got {v['Sex']!r}")

        if "Jaundice" in v and str(v["Jaundice"]).lower() not in _VALID_JAUNDICE:
            errors.append(
                f"Jaundice must be one of {sorted(_VALID_JAUNDICE)}, got {v['Jaundice']!r}"
            )

        if (
            "Family_mem_with_ASD" in v
            and str(v["Family_mem_with_ASD"]).lower() not in _VALID_FAMILY
        ):
            errors.append(
                f"Family_mem_with_ASD must be one of {sorted(_VALID_FAMILY)}, "
                f"got {v['Family_mem_with_ASD']!r}"
            )

        if (
            "Ethnicity" in v
            and str(v["Ethnicity"]).lower() not in _VALID_ETHNICITY
        ):
            errors.append(
                f"Ethnicity {v['Ethnicity']!r} is not in the list of known values. "
                "Use 'Others' if unsure."
            )

        if (
            "Who completed the test" in v
            and str(v["Who completed the test"]).lower() not in _VALID_WHO
        ):
            errors.append(
                f"'Who completed the test' must be one of {sorted(_VALID_WHO)}, "
                f"got {v['Who completed the test']!r}"
            )

        if errors:
            raise ValueError("; ".join(errors))

        return v


class PredictResponse(BaseModel):
    """Response body for the ``POST /ml/predict`` endpoint.

    Attributes:
        prediction: The model's predicted class label (e.g. ``"Yes"`` or ``"No"``).
        probability: Confidence score in ``[0, 1]``, or ``None`` if unavailable.
    """

    prediction: str = Field(
        ..., description="Predicted class label returned by the model."
    )
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0–1), if available.",
    )
