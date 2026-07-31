import asyncio
import os
from dotenv import load_dotenv
from project_x_py import ProjectX

load_dotenv()

print("API Key:", os.getenv("PROJECT_X_API_KEY", "NOT FOUND")[:15] + "...")
print("Username:", os.getenv("PROJECT_X_USERNAME", "NOT FOUND"))


async def test():
    try:
        async with ProjectX.from_env() as client:
            await asyncio.wait_for(client.authenticate(), timeout=15)
            print("SUCCESS! Connected.")
            print(f"Account: {client.account_info.name}")
    except asyncio.TimeoutError:
        print("ERROR: Connection timed out after 15 seconds")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

asyncio.run(test())
