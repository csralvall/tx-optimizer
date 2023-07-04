import logging
import logging.config

import structlog

def configure_loggers(
    log_level: str,
    loggers_to_disable: list[str] = [],
) -> None:
    """Setup and unify logging and structlog configuration.

    This function configures the processors and formatters that the loggers
    will use, disables the unused loggers and configures logging to use the
    same format as structlog.

    Args:
        log_level: The level at which to configure the loggers.
        loggers_to_disable: The names of the loggers to disable.
    """
    logging_processors = [
        # Add extra attributes of LogRecord objects to the event dictionary
        # so that values passed in the extra parameter of log methods pass
        # through to log output.
        structlog.stdlib.ExtraAdder(),
    ]

    common_processors = [
        # Add structlog context variables to log lines
        structlog.contextvars.merge_contextvars,
        # Add local thread variables to the log lines
        structlog.threadlocal.merge_threadlocal,
        # Add the name of the logger to the record
        structlog.stdlib.add_logger_name,
        # Adds the log level as a parameter of the log line
        structlog.stdlib.add_log_level,
        # Adds a timestamp for every log line
        structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S%z"),
        # Perform old school %-style formatting. on the log msg/event
        structlog.stdlib.PositionalArgumentsFormatter(),
        # If the log record contains a string in byte format, this will
        # automatically convert it into a utf-8 string
        structlog.processors.UnicodeDecoder(),
    ]

    pre_render_processors = [
        # Remove `_record` and `_from_structlog` keys from log event
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        # Unpack exception info in log entry
        structlog.processors.format_exc_info,
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            # Filter log entries by level
            structlog.stdlib.filter_by_level,
            *common_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        # Used to create wrapped loggers that are used for OUTPUT.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # The bound logger that you get back from get_logger().
        # This one imitates the API of `logging.Logger`.
        wrapper_class=structlog.stdlib.BoundLogger,
        # Effectively freeze configuration after creating the first bound logger.
        cache_logger_on_first_use=True,
    )

    # Configure logging (with structlog format)
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    *pre_render_processors,
                    # Render the final event dict in a key-value format.
                    structlog.dev.ConsoleRenderer(colors=True),
                ],
                "foreign_pre_chain": [
                    *common_processors,
                    *logging_processors,
                ],
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "colored",
            }
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            }
        },
    }

    logging.config.dictConfig(logging_config)

    # Disable loggers
    for log in loggers_to_disable:
        # For all those loggers recreated through other handlers (e.g.
        # middlewares) in the structured log, we clear the handlers and prevent
        # the logs to propagate to a logger higher up in the hierarchy
        logging.getLogger(log).handlers.clear()
        logging.getLogger(log).propagate = False
