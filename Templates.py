from pydantic import BaseModel
from typing import List, Optional

class WineBottle(BaseModel):
    wine_id: str
    wine_name: str
    type: str = "bottle"
    producer: str
    region: str
    bottle_name: str
    featured: bool = False

class VectorChunk(BaseModel):
    id: str
    text: str
    metadata: dict
    embedding: Optional[List[float]] = None