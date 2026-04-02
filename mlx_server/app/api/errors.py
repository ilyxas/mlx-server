from __future__ import annotations


class AppError(Exception):
    """Application-level error that maps to a structured API error response."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
