from pydantic import BaseModel, Field

class MetricsQuery(BaseModel):
    country: str = Field(..., examples=["Lithuania"])
    metric: str = Field(..., examples=["NEW_CASES_PER_100K"])
    start: str = Field(..., examples=["2021-01-01"])
    end: str = Field(..., examples=["2021-06-30"])

class CommentIn(BaseModel):
    country: str
    date: str
    metric: str
    user: str
    comment: str  #text

class MetricsRow(BaseModel):
    country: str
    date: str
    metric: str
    value: float
    annotation: dict[str, object] | None = None
