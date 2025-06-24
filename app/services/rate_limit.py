# app/services/rate_limit.py

import time
from typing import Any, Optional
import asyncio

# Import BaseRateLimiter จาก LangChain
from langchain_core.rate_limiters import BaseRateLimiter


# คลาสของเราสืบทอดจาก BaseRateLimiter
class RateLimiter(BaseRateLimiter):
    """A simple rate limiter that enforces a certain number of requests per minute."""

    def __init__(self, requests_per_minute: int):
        """
        Initializes the rate limiter.

        Args:
            requests_per_minute: The maximum number of requests allowed per minute.
        """
        self.requests_per_minute = requests_per_minute
        self.token_interval = 60.0 / self.requests_per_minute
        self.last_request_time: Optional[float] = None

    def acquire(self, **kwargs: Any) -> None:
        """
        Acquires a token, blocking if necessary until a token is available.
        This method is called by LangChain before making a request.
        """
        while True:
            current_time = time.monotonic()
            if (
                self.last_request_time is None
                or (current_time - self.last_request_time) >= self.token_interval
            ):
                self.last_request_time = current_time
                return
            
            sleep_time = self.token_interval - (current_time - self.last_request_time)
            time.sleep(sleep_time)

    async def aacquire(self, **kwargs: Any) -> None:
        """
        Asynchronously acquires a token, yielding control if necessary until a token is available.
        This method is called by LangChain for async operations.
        """
        while True:
            current_time = time.monotonic()
            if (
                self.last_request_time is None
                or (current_time - self.last_request_time) >= self.token_interval
            ):
                self.last_request_time = current_time
                return

            sleep_time = self.token_interval - (current_time - self.last_request_time)
            await asyncio.sleep(sleep_time)

    # --- FIX: เพิ่มเมธอดนี้เข้าไปเพื่อรองรับโค้ดส่วนที่เรียกใช้ .wait() ---
    def wait(self, **kwargs: Any) -> None:
        """An alias for the acquire method for compatibility with older patterns."""
        # ทำให้ wait() ทำงานเหมือน acquire() ทุกประการ
        self.acquire(**kwargs)