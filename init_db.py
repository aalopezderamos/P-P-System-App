from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime
import os

# Import from main
from main import User, Base, DATABASE_URL

print("🔧 Initializing database...")

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create all tables
Base.metadata.create_all(bind=engine)
print("✅ Database tables created")

# Create admin user if it doesn't exist
db = SessionLocal()
try:
    admin_username = os.getenv("ADMIN_USERNAME", "Aaron")
    admin_email = os.getenv("ADMIN_EMAIL", "aaron@predictandpour.com")
    admin_password = os.getenv("ADMIN_PASSWORD", "Allmight881")
    
    existing = db.query(User).filter(User.username == admin_username).first()
    
    if not existing:
        admin = User(
            username=admin_username,
            email=admin_email,
            password_hash=pwd_context.hash(admin_password),
            is_active=True,
            created_at=datetime.utcnow()
        )
        db.add(admin)
        db.commit()
        print(f"✅ Admin user '{admin_username}' created")
    else:
        print(f"ℹ️  Admin user '{admin_username}' already exists")
finally:
    db.close()

print("🚀 Database initialization complete")