from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime

# Database setup
DATABASE_URL = "sqlite:///./users.db"
Base = declarative_base()

# Define User model (same as in main.py)
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

# Create engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create tables
Base.metadata.create_all(bind=engine)

# Create database session
db = SessionLocal()

# Check if user already exists
existing = db.query(User).filter(User.username == "Aaron").first()
if existing:
    print("❌ User 'Aaron' already exists!")
    db.close()
    exit()

# Create your admin user
admin = User(
    username="Aaron",
    email="aaron@predictandpour.com",
    password_hash=pwd_context.hash("Allmight881"),
    is_active=True,
    created_at=datetime.utcnow()
)

db.add(admin)
db.commit()
print("✅ User 'Aaron' created successfully!")
print(f"   Username: Aaron")
print(f"   Email: aaron@predictandpour.com")
print(f"   Password: Allmight881")

db.close()