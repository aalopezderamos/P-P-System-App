from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime
import sys
sys.path.append('.')

# Import from main.py
from main import User, Base, DATABASE_URL

# Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create database session
db = SessionLocal()

# Define users to add
users_to_add = [
    {
        "username": "reviewer",
        "email": "reviewer@predictandpour.com",
        "password": "ChromeReview2026!"
    },
    {
        "username": "DaneJohnson",
        "email": "Dane.Johnson@silvereaglebev.com",
        "password": "Kingofthenorth"
    },
    {
        "username": "JacobAlvarez",
        "email": "Jacob.Alvarez@silvereaglebev.com",
        "password": "Jacobknowsball"
    }
]

# Add each user
for user_data in users_to_add:
    # Check if user already exists
    existing = db.query(User).filter(
        (User.username == user_data["username"]) | 
        (User.email == user_data["email"])
    ).first()
    
    if existing:
        print(f"⚠️  User '{user_data['username']}' already exists, skipping...")
        continue
    
    # Create new user
    new_user = User(
        username=user_data["username"],
        email=user_data["email"],
        password_hash=pwd_context.hash(user_data["password"]),
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    print(f"✅ User '{user_data['username']}' created successfully!")
    print(f"   Email: {user_data['email']}")
    print(f"   Password: {user_data['password']}")
    print()

db.close()
print("🎉 All users processed!")