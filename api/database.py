import pymongo
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "ObesityDB"
COLLECTION_NAME = "obesity_data"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
print(f"Connected to MongoDB: {DB_NAME}, Collection: {COLLECTION_NAME}")

def insert_data(data):
    
    if isinstance(data, list):
        result = collection.insert_many(data)
    else:
        result = collection.insert_one(data)
    
    print(f"Data inserted successfully: {len(result.inserted_ids)} records")

def fetch_all_data():
    data = list(collection.find({}, {"_id": 0}))
    return data

def clear_all_data():
    result = collection.delete_many({})
    return {"status": "success", "deleted_count": result.deleted_count}
