from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class MetricsQuery(BaseModel):
    country: str = Field(..., examples=["Lithuania"])
    metric: str  = Field(..., examples=["NEW_CASES_PER_100K"])
    start: str   = Field(..., examples=["2021-01-01"])  
    end: str     = Field(..., examples=["2021-06-30"])
    value: Optional[float] = None       
    sma7: Optional[float] = None        

class CommentIn(BaseModel):
    country: str
    date: str         
    metric: str
    user: str
    comment: str
    value: Optional[float] = None

class MetricsRow(BaseModel):
    country: str
    date: str
    metric: str
    value: float
    sma7: Optional[float] = None
    annotation: Optional[Dict[str, Any]] = None
