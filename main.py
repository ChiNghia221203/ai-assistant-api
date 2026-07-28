from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

list_item = [
    Item(name="Nghĩa",price=1.2,is_offer=True),
    Item(name="Khang",price=0.5,is_offer=False),
]
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.post("/item/", response_model= List[Item] ) 
def update_item_post(item: Item):
        list_item.insert(1,item)
        return list_item