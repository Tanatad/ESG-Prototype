# test_db.py

import asyncio
import os
from dotenv import load_dotenv
import beanie
import motor.motor_asyncio

# เราต้อง import Model ทั้งหมดที่ถูกลงทะเบียนไว้ใน Beanie
from app.models.esg_question_model import ESGQuestion, PipelineRun
from app.models.clustering_model import Cluster


async def test_database_query():
    """
    สคริปต์สำหรับทดสอบการเชื่อมต่อ DB และ Query ข้อมูลล่าสุดจาก pipeline_runs โดยตรง
    """
    print("--- Starting Database Test Script ---")

    # 1. โหลดค่าจากไฟล์ .env
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URL")
    mongo_db_name = os.getenv("MONGO_DB_NAME")

    if not mongo_uri or not mongo_db_name:
        print("!!! ERROR: กรุณาตรวจสอบว่ามี MONGO_URL และ MONGO_DB_NAME ในไฟล์ .env")
        return

    print(f"Connecting to MongoDB database '{mongo_db_name}'...")

    # 2. เชื่อมต่อ Database และลงทะเบียน Beanie Models
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        database = client[mongo_db_name]
        await beanie.init_beanie(database=database, document_models=[ESGQuestion, Cluster, PipelineRun])
        print("Beanie initialized successfully.")
    except Exception as e:
        print(f"!!! ERROR during Beanie initialization: {e}")
        return

    # 3. ทำการ Query เพื่อหาข้อมูลล่าสุด
    print("\n--- Performing the query to find the latest run ---")
    print("Running: PipelineRun.find_all(sort=[('run_timestamp', -1)]).limit(1).first_or_none()")
    
    try:
        latest_run = await PipelineRun.find_all(
            sort=[("run_timestamp", -1)]
        ).limit(1).first_or_none()

        # 4. พิมพ์ผลลัพธ์ที่ได้
        print("\n--- Query Result ---")
        if latest_run:
            print("✅ SUCCESS: Found a document!")
            print(f"   Document ID: {latest_run.id}")
            print(f"   Timestamp: {latest_run.run_timestamp}")
            print(f"   Verification Status: {latest_run.verification_summary.get('status')}")
        else:
            print("❌ FAILED: Query returned None. No document was found.")

    except Exception as e:
        print(f"💥 An exception occurred during the query: {e}")

    print("\n--- Database Test Script Finished ---")


if __name__ == "__main__":
    asyncio.run(test_database_query())