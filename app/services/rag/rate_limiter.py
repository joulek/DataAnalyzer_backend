"""Limitation du débit : espacement des appels au LLM et quota par utilisateur.

- Espacement global : deux appels au LLM ne partent jamais plus vite que MIN_INTERVAL_SECONDS.
- Quota par client : un client ne peut pas poser plus de MAX_REQUESTS_PER_WINDOW questions
  dans une fenêtre glissante de WINDOW_SECONDS.

Le stockage est en mémoire : il est remis à zéro au redémarrage du serveur, et ne fonctionne
que pour un seul processus uvicorn. Pour plusieurs instances, il faudrait Redis.
"""

import threading
import time
from collections import defaultdict, deque

# --- Réglages (à ajuster selon votre quota Groq) ---
MIN_INTERVAL_SECONDS = 0.5        # délai minimal entre deux appels au LLM
MAX_REQUESTS_PER_WINDOW = 10      # questions autorisées par client...
WINDOW_SECONDS = 600              # ...dans une fenêtre de 10 minutes


class RateLimitExceeded(Exception):
    """Levée quand un client a dépassé son quota."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Trop de questions. Réessayez dans {retry_after} secondes.")


class _GlobalSpacer:
    """Garantit un intervalle minimal entre deux appels, même venant de threads différents."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait_turn(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


class _ClientQuota:
    """Fenêtre glissante par identifiant client."""

    def __init__(self, max_requests: int, window: float):
        self._max = max_requests
        self._window = window
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client_id]
            # On retire les appels sortis de la fenêtre
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._max:
                retry_after = int(self._window - (now - hits[0])) + 1
                raise RateLimitExceeded(retry_after)
            hits.append(now)


_spacer = _GlobalSpacer(MIN_INTERVAL_SECONDS)
_quota = _ClientQuota(MAX_REQUESTS_PER_WINDOW, WINDOW_SECONDS)


def check_client_quota(client_id: str) -> None:
    """À appeler dès qu'une question arrive. Lève RateLimitExceeded si le quota est atteint."""
    _quota.check(client_id)


def wait_before_llm_call() -> None:
    """À appeler juste avant chaque appel au LLM. Attend si nécessaire."""
    _spacer.wait_turn()