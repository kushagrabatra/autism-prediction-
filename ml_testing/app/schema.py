from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

example_features = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
    "A6": 0, "A7": 0, "A8": 0, "A9": 0, "A10": 0,
    "Age_Mons": 36, "Qchat-10-Score": 6,
    "Sex": "m", "Ethnicity": "White European",
    "Jaundice": "no", "Family_mem_with_ASD": "no",
    "Who completed the test": "family member"
}

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., example=example_features)

class PredictResponse(BaseModel):
    prediction: str
    probability: Optional[float] = None
