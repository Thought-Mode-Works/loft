"""
Decorators for automatic logging of LLM interactions and symbolic operations.

These decorators enable traceability without cluttering business logic.
"""

import functools
from typing import Callable, Any
from datetime import datetime
from .logger import get_loft_logger, log_llm_interaction, log_symbolic_operation


def track_llm_call(
    model_param: str = "model",
    prompt_param: str = "prompt",
    capture_response: bool = True,
) -> Callable:
    """
    Decorator to track LLM API calls.

    Logs the full request and response with metadata for traceability.

    Args:
        model_param: Name of the parameter containing the model name
        prompt_param: Name of the parameter containing the prompt
        capture_response: Whether to log the response

    Example:
        >>> @track_llm_call()
        ... def query_llm(prompt: str, model: str = "claude-3"):
        ...     return call_api(prompt, model)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_loft_logger("llm")

            # Extract model and prompt from args/kwargs
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            model = bound_args.arguments.get(model_param, "unknown")
            prompt = bound_args.arguments.get(prompt_param, "")

            # Truncate prompt for logging (first 500 chars)
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt

            # Log request
            request_id = datetime.utcnow().timestamp()
            log_llm_interaction(
                log,
                event="request",
                model=model,
                prompt_length=len(prompt),
                prompt_preview=prompt_preview,
                request_id=request_id,
                function=func.__name__,
            )

            try:
                # Call the function
                result = func(*args, **kwargs)

                # Log response if enabled
                if capture_response:
                    response_preview = ""
                    response_length = 0

                    if isinstance(result, str):
                        response_preview = result[:500] + "..." if len(result) > 500 else result
                        response_length = len(result)
                    elif isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        response_preview = content[:500] + "..." if len(content) > 500 else content
                        response_length = len(content)

                    log_llm_interaction(
                        log,
                        event="response",
                        model=model,
                        response_length=response_length,
                        response_preview=response_preview,
                        request_id=request_id,
                        function=func.__name__,
                        success=True,
                    )

                return result

            except Exception as e:
                # Log error
                log_llm_interaction(
                    log,
                    event="error",
                    model=model,
                    request_id=request_id,
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                    success=False,
                )
                raise

        return wrapper

    return decorator


def track_symbolic_operation(operation_type: str) -> Callable:
    """
    Decorator to track symbolic core operations.

    Logs rule modifications, queries, and other symbolic operations.

    Args:
        operation_type: Type of operation (e.g., "rule_add", "query", "validate")

    Example:
        >>> @track_symbolic_operation("rule_add")
        ... def add_rule(rule: str):
        ...     self.rules.append(rule)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_loft_logger("symbolic_core")

            # Extract relevant parameters
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Log operation start
            operation_id = datetime.utcnow().timestamp()
            log_symbolic_operation(
                log,
                operation=operation_type,
                operation_id=operation_id,
                function=func.__name__,
                arguments={
                    k: str(v)[:100] for k, v in bound_args.arguments.items()
                },  # Truncate args
            )

            try:
                result = func(*args, **kwargs)

                # Log operation completion
                log_symbolic_operation(
                    log,
                    operation=f"{operation_type}_complete",
                    operation_id=operation_id,
                    function=func.__name__,
                    success=True,
                )

                return result

            except Exception as e:
                # Log operation failure
                log_symbolic_operation(
                    log,
                    operation=f"{operation_type}_error",
                    operation_id=operation_id,
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                    success=False,
                )
                raise

        return wrapper

    return decorator


def track_validation(validation_type: str) -> Callable:
    """
    Decorator to track validation operations.

    Logs validation checks with structured results.

    Args:
        validation_type: Type of validation (e.g., "syntax", "semantic", "fidelity")

    Example:
        >>> @track_validation("syntax")
        ... def validate_syntax(program: str) -> bool:
        ...     return check_syntax(program)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_loft_logger("validation")

            validation_id = datetime.utcnow().timestamp()

            # Extract input for logging
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            try:
                result = func(*args, **kwargs)

                # Determine if validation passed
                passed = bool(result) if isinstance(result, bool) else True

                # Log validation result
                from .logger import log_validation_result

                log_validation_result(
                    log,
                    validation_type=validation_type,
                    result=passed,
                    validation_id=validation_id,
                    function=func.__name__,
                )

                return result

            except Exception as e:
                # Log validation error
                log_validation_result(
                    log,
                    validation_type=validation_type,
                    result=False,
                    validation_id=validation_id,
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator


def track_meta_reasoning(event_type: str) -> Callable:
    """
    Decorator to track meta-reasoning events.

    Logs self-assessment, strategy changes, and learning events.

    Args:
        event_type: Type of meta-reasoning event

    Example:
        >>> @track_meta_reasoning("self_assessment")
        ... def assess_confidence(predictions: List) -> float:
        ...     return compute_confidence(predictions)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_loft_logger("meta_reasoning")

            event_id = datetime.utcnow().timestamp()

            try:
                result = func(*args, **kwargs)

                # Log meta-reasoning event
                from .logger import log_meta_reasoning

                log_meta_reasoning(
                    log,
                    event=event_type,
                    event_id=event_id,
                    function=func.__name__,
                    result=str(result)[:200] if result is not None else None,
                )

                return result

            except Exception as e:
                log.error(
                    f"Meta-reasoning error: {event_type}",
                    event_type=event_type,
                    event_id=event_id,
                    function=func.__name__,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator


def performance_monitor(threshold_ms: float = 1000.0) -> Callable:
    """
    Decorator to monitor function performance.

    Logs warning if execution exceeds threshold.

    Args:
        threshold_ms: Warning threshold in milliseconds

    Example:
        >>> @performance_monitor(threshold_ms=500)
        ... def expensive_operation():
        ...     time.sleep(1)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = get_loft_logger("system")

            import time

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if elapsed_ms > threshold_ms:
                    log.warning(
                        f"Performance threshold exceeded: {func.__name__}",
                        function=func.__name__,
                        elapsed_ms=elapsed_ms,
                        threshold_ms=threshold_ms,
                    )
                else:
                    log.debug(
                        f"Function executed: {func.__name__}",
                        function=func.__name__,
                        elapsed_ms=elapsed_ms,
                    )

                return result

            except Exception:
                elapsed_ms = (time.time() - start_time) * 1000
                log.debug(
                    f"Function failed: {func.__name__}",
                    function=func.__name__,
                    elapsed_ms=elapsed_ms,
                )
                raise

        return wrapper

    return decorator
