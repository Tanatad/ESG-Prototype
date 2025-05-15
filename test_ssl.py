import aiohttp
import asyncio

async def test_ssl_in_app():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get('https://www.google.com') as response:
                print('Status:', response.status)
        except Exception as e:
            print('SSL error in app context:', e)

asyncio.run(test_ssl_in_app())
