from pydantic import BaseModel
from typing import List

class Wine(BaseModel):
    id: str
    text: str
    embedding: List[float]