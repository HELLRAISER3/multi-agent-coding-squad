from src.api.agent.agent import StreamingAgentExecutor
import asyncio
import time

class SessionStore:
    def __init__(self, ttl_seconds: int = 1800):
        self._store: dict[str, tuple[StreamingAgentExecutor, float]] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> StreamingAgentExecutor:
        async with self._lock:
            self._evict_expired()
            if session_id not in self._store:
                self._store[session_id] = (StreamingAgentExecutor(), time.time())
            executor, _ = self._store[session_id]
            self._store[session_id] = (executor, time.time())  # refresh TTL
            return executor

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]

session_store = SessionStore()