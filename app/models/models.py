"""
SQLAlchemy models for the recommendation system
Converted from Sequelize models
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Table
from sqlalchemy.orm import relationship

from app.core.database import Base


# Association table for User-Service many-to-many relationship
class UserService(Base):
    """
    Association table for tracking user-service relationships
    Stores the number of calls each user made to each service
    """
    __tablename__ = "UserServices"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("Users.id"), nullable=False)
    service_id = Column(Integer, ForeignKey("Services.id"), nullable=False)
    number_of_calls = Column(Integer, default=0)
    
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="user_services")
    service = relationship("Service", back_populates="user_services")


class Call(Base):
    """
    Call model - stores service call history
    """
    __tablename__ = "Calls"
    
    id = Column(Integer, primary_key=True, unique=True)
    classname = Column(String)
    console_output = Column(Text)
    created_by = Column(String)
    created_on = Column(DateTime)
    edited_by = Column(String)
    edited_on = Column(DateTime)
    end_time = Column(DateTime)
    error_output = Column(Text)
    input = Column(Text)
    input_data = Column(Text)
    input_params = Column(Text)
    is_deleted = Column(String)
    mid = Column(Integer, index=True)  # Service ID
    os_pid = Column(Integer)
    owner = Column(String, index=True)  # User ID
    result = Column(Text)
    start_time = Column(DateTime, index=True)
    status = Column(String, index=True)
    
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Service(Base):
    """
    Service model - stores service metadata
    """
    __tablename__ = "Services"
    
    id = Column(Integer, primary_key=True, unique=True)
    name = Column(String)
    subject = Column(String)
    type = Column(String, index=True)
    description = Column(Text)
    number_of_calls = Column(Integer)
    actionview = Column(String)
    actionmodify = Column(String)
    map_reduce_specification = Column(String)
    params = Column(JSON)
    js_body = Column(Text)
    wpsservers = Column(JSON)
    wpsmethod = Column(String)
    status = Column(String)
    output_params = Column(JSON)
    wms_link = Column(String)
    wms_layer_name = Column(String)
    is_deleted = Column(String)
    created_by = Column(String)
    edited_by = Column(String)
    edited_on = Column(DateTime)
    created_on = Column(DateTime)
    classname = Column(String)
    
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user_services = relationship("UserService", back_populates="service")


class Composition(Base):
    """
    Composition model - stores service composition workflows
    """
    __tablename__ = "Compositions"
    
    id = Column(Text, primary_key=True, unique=True)
    nodes = Column(JSON, nullable=False)  # Array of composition nodes
    links = Column(JSON, nullable=False)  # Array of links between nodes
    
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TableComposition(Base):
    """
    TableComposition model - minimal table-only compositions.
    Contains only composition id and ordered table sequence.
    """
    __tablename__ = "TableCompositions"

    id = Column(Text, primary_key=True, unique=True)
    table_ids = Column(JSON, nullable=False)
    # Backward-compatible placeholders required by existing DB schema.
    # Logical payload is table_ids; these are stored as empty arrays.
    nodes = Column(JSON, nullable=False, default=list)
    links = Column(JSON, nullable=False, default=list)

    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Dataset(Base):
    """
    Dataset model - stores dataset metadata
    """
    __tablename__ = "Datasets"
    
    id = Column(Integer, primary_key=True, unique=True)
    guid = Column(String)
    
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class User(Base):
    """
    User model - stores user information
    """
    __tablename__ = "Users"
    
    id = Column(String, primary_key=True, unique=True)
    
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user_services = relationship("UserService", back_populates="user")

