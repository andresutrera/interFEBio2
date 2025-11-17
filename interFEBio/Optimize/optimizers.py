"""Adapters that bridge scipy optimisers to the engine interface."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import numpy as np

BoundsLike = Sequence[tuple[float, float]] | tuple[np.ndarray, np.ndarray] | None
Callback = Callable[[np.ndarray, float], None]


class OptimizerAdapter:
    """Abstract interface implemented by optimiser adapters."""

    def minimize(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray], np.ndarray] | None,
        phi0: np.ndarray,
        bounds: BoundsLike,
        callbacks: Iterable[Callback] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Minimise the objective using the configured optimiser."""
        """Minimise the objective using the configured optimiser."""
        raise NotImplementedError

    @staticmethod
    def build(name: str, options: Mapping[str, Any] | None) -> "OptimizerAdapter":
        opts = dict(options or {})
        key = name.lower()
        if key == "least_squares":
            return ScipyLeastSquaresAdapter(**opts)
        if key == "minimize":
            method = opts.pop("method", "L-BFGS-B")
            return ScipyMinimizeAdapter(method=method, **opts)
        raise ValueError(f"Unsupported optimizer: {name}")


class ScipyLeastSquaresAdapter(OptimizerAdapter):
    """Adapter that wraps :func:`scipy.optimize.least_squares`."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def minimize(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray], np.ndarray] | None,
        phi0: np.ndarray,
        bounds: BoundsLike,
        callbacks: Iterable[Callback] | None = None,
    ) -> tuple[np.ndarray, dict]:
        import scipy.optimize

        cb_list = list(callbacks or [])

        def _callback(x, f=None, accepted=None):
            if f is None and hasattr(x, "x") and hasattr(x, "fun"):
                vec = np.asarray(x.x, dtype=float)
                resid = np.asarray(x.fun, dtype=float)
            else:
                vec = np.asarray(x, dtype=float)
                resid = np.asarray(f if f is not None else fun(vec), dtype=float)
            cost = 0.5 * float(np.dot(resid, resid))
            for cb in cb_list:
                cb(vec, cost)

        if bounds is None:
            lower = np.full_like(phi0, -np.inf, dtype=float)
            upper = np.full_like(phi0, np.inf, dtype=float)
        elif (
            isinstance(bounds, tuple)
            and len(bounds) == 2
            and all(isinstance(b, np.ndarray) for b in bounds)
        ):
            lower = np.asarray(bounds[0], dtype=float)
            upper = np.asarray(bounds[1], dtype=float)
        else:
            seq_bounds = cast(Sequence[tuple[float, float]], bounds)
            lower = np.asarray([b[0] for b in seq_bounds], dtype=float)
            upper = np.asarray([b[1] for b in seq_bounds], dtype=float)
        lsq_bounds = (lower, upper)

        def _logged_fun(x: np.ndarray) -> np.ndarray:
            vec = np.asarray(x, dtype=float)
            resid = fun(vec)
            # print("fun result", resid)
            return resid

        if jac is None:
            wrapped_jac: str | Callable[[np.ndarray], np.ndarray] = "2-point"
        else:

            def _logged_jac(x: np.ndarray) -> np.ndarray:
                vec = np.asarray(x, dtype=float)
                mat = jac(vec)
                # print("jac result", mat)
                return mat

            wrapped_jac = _logged_jac

        result = scipy.optimize.least_squares(  # type: ignore[attr-defined]
            _logged_fun,
            phi0,
            jac=wrapped_jac,
            bounds=lsq_bounds,
            callback=_callback if cb_list else None,
            **self.kwargs,
        )
        if cb_list:
            cost = 0.5 * float(np.dot(result.fun, result.fun))
            for cb in cb_list:
                cb(result.x, cost)
        return result.x, result.__dict__


class ScipyMinimizeAdapter(OptimizerAdapter):
    """Adapter that wraps :func:`scipy.optimize.minimize`."""

    def __init__(self, method: str = "L-BFGS-B", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def minimize(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray], np.ndarray] | None,
        phi0: np.ndarray,
        bounds: BoundsLike,
        callbacks: Iterable[Callback] | None = None,
    ) -> tuple[np.ndarray, dict]:
        import scipy.optimize

        if jac is None:
            raise ValueError("ScipyMinimizeAdapter requires a gradient (jacobian)")

        def objective(x: np.ndarray) -> float:
            r = fun(x)
            return 0.5 * float(np.dot(r, r))

        def grad(x: np.ndarray) -> np.ndarray:
            r = fun(x)
            J = jac(x)
            return cast(np.ndarray, J.T @ r)

        cb_list = list(callbacks or [])

        def _callback(xk):
            vec = np.asarray(xk, dtype=float)
            resid = fun(vec)
            cost = 0.5 * float(np.dot(resid, resid))
            for cb in cb_list:
                cb(vec, cost)

        if (
            isinstance(bounds, tuple)
            and len(bounds) == 2
            and all(isinstance(b, np.ndarray) for b in bounds)
        ):
            lower = np.asarray(bounds[0], dtype=float)
            upper = np.asarray(bounds[1], dtype=float)
            min_bounds: Sequence[tuple[float, float]] | None = list(
                zip(lower.tolist(), upper.tolist(), strict=True)
            )
        else:
            min_bounds = cast(Sequence[tuple[float, float]] | None, bounds)

        result = scipy.optimize.minimize(  # type: ignore[attr-defined]
            objective,
            phi0,
            jac=grad,
            method=self.method,
            bounds=min_bounds,
            callback=_callback if cb_list else None,
            **self.kwargs,
        )
        if cb_list:
            cost = 0.5 * float(np.dot(fun(result.x), fun(result.x)))
            for cb in cb_list:
                cb(result.x, cost)
        return result.x, result.__dict__
