import asyncio
import sys
import os
from pprint import pprint

# Add backend to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.repositories.mongo_client import MongoRepository

async def test_database():
    print("🔌 Testing MongoDB Connection...")
    repo = MongoRepository()
    
    try:
        # 1. Connect
        await repo.connect()
        print("✅ Connection verified!")
        
        # 2. Check Health
        print("\n🏥 Checking Database Health...")
        health = await repo.health_check()
        pprint(health)
        
        if health['status'] != 'healthy':
            print("❌ Database is unhealthy. Aborting fetch test.")
            return

        # 3. Fetch a sample using raw collection access
        print("\n📄 Fetching one sample document (raw)...")
        if repo.collection is not None:
            sample = await repo.collection.find_one({}, {'_id': 0, 'College Name': 1, 'Location': 1, 'Fees': 1})
            if sample:
                print("✅ Found a document:")
                pprint(sample)
            else:
                print("⚠️ No documents found in collection.")
                
        # 4. Fetch using search method (retrieve all/limit)
        print("\n🔎 Testing search (Limit 3)...")
        # Empty query should return documents if logic allows, or we search for "Kathmandu"
        results = await repo.search_colleges({"Location": "Kathmandu"}) 
        # Note: Depending on search_colleges implementation, this input format might vary.
        # Let's try to query just by using find directly first if search_colleges is complex.
        # Looking at search_colleges signature: query_params: Dict[str, Any]
        
        if results:
            print(f"✅ Search returned {len(results)} results.")
            print("First result:")
            pprint(results[0])
        else:
            print("⚠️ Search returned no results (This might be expected if 'Kathmandu' is not in DB).")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await repo.disconnect()

if __name__ == "__main__":
    asyncio.run(test_database())
