from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class HeartInput(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: int = Field(description="1=Male,0=Female")
    cp: int = Field(ge=0, le=3)
    trtbps: int = Field(ge=70, le=250)
    chol: int = Field(ge=80, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalachh: int = Field(ge=60, le=220)
    exng: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slp: int = Field(ge=0, le=2)
    caa: int = Field(ge=0, le=3)
    thall: int = Field(ge=0, le=3)

    def to_feature_list(self) -> List[float]:
        return [self.age, self.sex, self.cp, self.trtbps, self.chol, self.fbs,
                self.restecg, self.thalachh, self.exng, self.oldpeak, self.slp, self.caa, self.thall]

class PredictRequest(BaseModel):
    data: List[HeartInput]
    client: Optional[str] = None

class PredictResponseItem(BaseModel):
    risk_percent: float
    risk_level: str
    probabilities: Dict[str, float]

class PredictResponse(BaseModel):
    results: List[PredictResponseItem]
    model_version: str

class TrainRequest(BaseModel):
    dataset_path: str
    target_column: str = "target"

class TrainResponse(BaseModel):
    model_version: str
    metrics: Dict[str, Optional[float]]
    features: List[str]
    class_distribution: Dict[str, int]
    confusion_matrix: List[List[int]]
