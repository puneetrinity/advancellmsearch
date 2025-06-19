from pydantic import BaseModel

class ChatResponse(BaseModel):
    message: str = "Dummy chat response"
    status: str = "success"
    error_code: str = None

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str = "Dummy error response"
    error_code: str = None
    error: str = None
