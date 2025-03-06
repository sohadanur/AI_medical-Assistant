#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create directories if they don't exist
mkdir -p templates
mkdir -p static
mkdir -p uploads

# Run the FastAPI application
uvicorn main:app --reload