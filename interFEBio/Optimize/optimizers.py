from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np


class OptimizerAdapter:
    def minimize(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Optional[Callable[[np.ndarray], np.ndarray]],
        phi0: np.ndarray,
        bounds: Optional[Sequence[Tuple[float, float]]],
        callbacks: Optional[Iterable[Callable[[np.ndarray, np.ndarray], None]]] = None,
    ) -> Tuple[np.ndarray, dict]:
        raise NotImplementedError


class ScipyLeastSquaresAdapter(OptimizerAdapter):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def minimize(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Optional[Callable[[np.ndarray], np.ndarray]],
        phi0: np.ndarray,
        bounds: Optional[Sequence[Tuple[float, float]]],
        callbacks: Optional[Iterable[Callable[[np.ndarray, np.ndarray], None]]] = None,
    ) -> Tuple[np.ndarray, dict]:
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

        result = scipy.optimize.least_squares(  # type: ignore[attr-defined]
            fun,
            phi0,
            jac=jac if jac is not None else "2-point",
            bounds=bounds if bounds is not None else (-np.inf, np.inf),
            callback=_callback if cb_list else None,
            **self.kwargs,
        )
        if cb_list:
            cost = 0.5 * float(np.dot(result.fun, result.fun))
            for cb in cb_list:
                cb(result.x, cost)
        return result.x, result.__dict__


class ScipyMinimizeAdapter(OptimizerAdapter):
    def __init__(self, method: str = "L-BFGS-B", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def minimize(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Optional[Callable[[np.ndarray], np.ndarray]],
        phi0: np.ndarray,
        bounds: Optional[Sequence[Tuple[float, float]]],
        callbacks: Optional[Iterable[Callable[[np.ndarray, np.ndarray], None]]] = None,
    ) -> Tuple[np.ndarray, dict]:
        import scipy.optimize

        if jac is None:
            raise ValueError("ScipyMinimizeAdapter requires a gradient (jacobian)")

        def objective(x: np.ndarray) -> float:
            r = fun(x)
            return 0.5 * float(np.dot(r, r))

        def grad(x: np.ndarray) -> np.ndarray:
            r = fun(x)
            J = jac(x)
            return J.T @ r

        cb_list = list(callbacks or [])

        def _callback(xk):
            vec = np.asarray(xk, dtype=float)
            resid = fun(vec)
            cost = 0.5 * float(np.dot(resid, resid))
            for cb in cb_list:
                cb(vec, cost)

        result = scipy.optimize.minimize(  # type: ignore[attr-defined]
            objective,
            phi0,
            jac=grad,
            method=self.method,
            bounds=bounds,
            callback=_callback if cb_list else None,
            **self.kwargs,
        )
        if cb_list:
            cost = 0.5 * float(np.dot(fun(result.x), fun(result.x)))
            for cb in cb_list:
                cb(result.x, cost)
        return result.x, result.__dict__
