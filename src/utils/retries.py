import time
from collections.abc import Callable

import structlog

LOGGER = structlog.stdlib.get_logger(__name__)


class retry:
    def __init__(
        self,
        tries: int = 3,
        delay: int = 2,
        backoff_factor: int = 3,
        exceptions_to_retry: tuple[type[Exception], ...] = (),
        exceptions_to_raise: tuple[type[Exception], ...] = (),
    ) -> None:
        self.pending_tries = self.tries = tries
        self.current_delay = self.delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions_to_retry = (*exceptions_to_retry, Exception)
        self.exceptions_to_raise = exceptions_to_raise
        self._last_exception: Exception | None = None

    @property
    def retriable(self) -> bool:
        return self.pending_tries > 1

    def should_raise_exception(self, exception_raised: Exception) -> bool:
        return any(
            isinstance(exception_raised, exception_class)
            for exception_class in self.exceptions_to_raise
        )

    def process_retry(self, exception_raised: Exception) -> None:
        LOGGER.debug(
            f"{exception_raised}, retrying in {self.current_delay} seconds."
        )

        time.sleep(self.current_delay)
        self.pending_tries -= 1
        self.current_delay *= self.backoff_factor

    def reset(self) -> None:
        self.current_delay: int = self.delay
        self.pending_tries: int = self.tries

    def __call__(self, f: Callable) -> Callable:
        def f_retry(*args, **kwargs):
            self.reset()
            while self.retriable:
                try:
                    return f(*args, **kwargs)
                except self.exceptions_to_retry as e:
                    self._last_exception = e
                    if self.should_raise_exception(e):
                        raise
                    self.process_retry(e)
            return f(*args, **kwargs)

        f_retry.__name__ = f.__name__
        return f_retry
