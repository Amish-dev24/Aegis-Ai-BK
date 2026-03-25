"""
Authentication endpoints.
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash,
    require_admin
)
from jose import JWTError, jwt
from app.config import settings
from app.schemas.user import Token, UserResponse, UserCreate, UserSignup
from app.models.user import User, Role
from app.models.company import Company
from app.models.audit_log import create_audit_log

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login endpoint."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token (30 min) and refresh token (2 hours)
    token_data = {"sub": user.username, "role": user.role.value}
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_refresh_token(
        data=token_data,
        expires_delta=timedelta(hours=settings.REFRESH_TOKEN_EXPIRE_HOURS)
    )

    # Log authentication
    db.add(create_audit_log(
        request, user.id, "login", "user", user.id,
        {"username": user.username, "role": user.role.value}
    ))
    db.commit()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """Use a valid refresh token to get a new access token.

    - Refresh token is valid for 2 hours after login.
    - Returns a new access token (30 min) without requiring re-login.
    - After 2 hours, the user must login again.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token. Please login again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        if username is None or token_type != "refresh":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Verify user still exists and is active
    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception

    # Issue new access token (30 min) and keep same refresh token
    token_data = {"sub": user.username, "role": user.role.value}
    new_access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {
        "access_token": new_access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserSignup,
    db: Session = Depends(get_db),
):
    """Public signup — no login required.

    Creates a new user with role=viewer and is_active=False.
    An admin must activate the user before they can login.
    """
    # Check if username already taken
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    # Check if email already taken
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Validate company exists (if provided)
    if user_data.company_id:
        company = db.query(Company).filter(Company.id == user_data.company_id).first()
        if not company:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company not found"
            )

    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        phone_number=user_data.phone_number,
        role=Role.VIEWER,
        company_id=user_data.company_id,
        is_active=False,  # Admin must activate
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Register a new user (admin only). Company admins can only create users in their company."""
    # Company admins cannot create Aegis AI admins
    if current_user.role != Role.AEGIS_ADMIN and user_data.role == Role.AEGIS_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create AEGIS_ADMIN user"
        )
    
    # Check if user exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Determine company_id
    if current_user.role == Role.AEGIS_ADMIN:
        # Aegis AI admins can create users for any company or without company
        company_id = user_data.company_id
    else:
        # Company admins create users in their own company
        company_id = current_user.company_id
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        phone_number=user_data.phone_number,
        role=user_data.role,
        company_id=company_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Log user creation
    db.add(create_audit_log(
        request, current_user.id, "create_user", "user", db_user.id,
        {"username": db_user.username, "role": db_user.role.value, "company_id": db_user.company_id}
    ))
    db.commit()

    return db_user


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user

