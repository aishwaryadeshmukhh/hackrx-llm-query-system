# Token endpoint for generating API tokens
# In a real application, you would implement proper user authentication here
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

# Router for auth endpoints
router = APIRouter()

# Hardcoded token - in a real app, this would be in a database
API_TOKEN = "552a90e441d8b2a0c195b5425dd982e0e71292568a08d2facf1ebc9434c1bcd0"

# Token response model
class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In a real app, you would validate username/password against a database
    # This is a simplified example with hardcoded credentials
    if form_data.username == "hackathon" and form_data.password == "hackrxpassword":
        return {"access_token": API_TOKEN, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Incorrect username or password")
