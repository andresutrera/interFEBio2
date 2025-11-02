import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger as _logger

if TYPE_CHECKING:
    from loguru import Logger  # only for type checking


class Log:
    _instance: Optional["Log"] = None

    def __new__(cls: type["Log"], *args: Any, **kwargs: Any) -> "Log":
        """
        Create or return the singleton instance of the Log class.

        On the first call, it creates a new instance and configures the logger.
        Subsequent calls can optionally trigger a reconfiguration when new arguments
        or keyword options are provided.

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._configure(*args, **kwargs)
        elif args or kwargs:
            cls._instance._configure(*args, **kwargs)
        return cls._instance

    def _configure(
        self,
        log_file: str | Path | None = None,
        level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "10 days",
        debug_mode: bool = False,
    ) -> None:
        """
        Configure the underlying loguru logger.

        This sets up console and optional file logging with specified levels,
        rotation, retention, and debug mode.

        """
        _logger.remove()
        console_level = "DEBUG" if debug_mode else level

        _logger.add(
            sys.stdout,
            level=console_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
            enqueue=False,
        )

        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            _logger.add(
                path,
                level=level,
                rotation=rotation,
                retention=retention,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} - {message}",
                enqueue=False,
                mode="w",  # overwrite the log file on each run
            )

    @property
    def logger(self) -> "Logger":  # use string literal here
        """
        Return the underlying loguru logger instance.

        This property provides access to the configured singleton logger.
        All logging calls should be made through this object.

        """
        return _logger
