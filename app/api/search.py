from fastapi import APIRouter

router = APIRouter()

@router.get("/search/dummy")
def dummy_search():
    return {"message": "Dummy search endpoint"}
