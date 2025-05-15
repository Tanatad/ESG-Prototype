from tenacity import retry, stop_after_attempt, wait_fixed
from functools import wraps
import asyncio

def retry_on_exception(func=None, *, attempts: int = 3, delay: int = 1):
    """
    Decorator to retry a coroutine up to a specified number of times with a fixed delay if an exception occurs.

    Parameters:
        func (Optional[Callable]): The function to wrap. Allows decorator to work with or without parameters.
        attempts (int): Number of retry attempts (default=3).
        delay (int): Delay in seconds between retries (default=1).
    """
    if func is None:
        def wrapper_with_params(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                for attempt in range(attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if attempt < attempts - 1:
                            await asyncio.sleep(delay)
                        else:
                            raise e
            return wrapper
        return wrapper_with_params
    else:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < attempts - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise e
        return wrapper
