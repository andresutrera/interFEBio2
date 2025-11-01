from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Protocol, Any
import numpy as np

from .feb_bindings import FebBuilder, FebTemplate, BuildContext

ArrayLike = Sequence[float]


class TaskContext(dict):
    """Mutable context passed between tasks."""


class Task(Protocol):
    def run(self, context: TaskContext) -> TaskContext:
        ...


@dataclass
class CallableTask:
    fn: Callable[[TaskContext], TaskContext]

    def run(self, context: TaskContext) -> TaskContext:
        return self.fn(context)


@dataclass
class PreprocessTask(CallableTask):
    pass


@dataclass
class PostprocessTask(CallableTask):
    pass


@dataclass
class CustomScriptTask(CallableTask):
    pass


@dataclass
class TaskPipeline:
    pre_tasks: List[Task] = field(default_factory=list)
    post_tasks: List[Task] = field(default_factory=list)

    def add_pre(self, task: Task) -> None:
        self.pre_tasks.append(task)

    def add_post(self, task: Task) -> None:
        self.post_tasks.append(task)

    def run_pre(self, context: TaskContext) -> TaskContext:
        for task in self.pre_tasks:
            context = task.run(context)
        return context

    def run_post(self, context: TaskContext) -> TaskContext:
        for task in self.post_tasks:
            context = task.run(context)
        return context


class SimulationReader(Protocol):
    def __call__(self, xplt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        ...


@dataclass
class SimulationAdapter:
    reader: SimulationReader

    def read(self, xplt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        return self.reader(Path(xplt_path))


WeightFunction = Callable[[ArrayLike], np.ndarray]


@dataclass
class ExperimentSeries:
    x: np.ndarray
    y: np.ndarray
    weight: Optional[WeightFunction] = None

    def weighted(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.weight is None:
            return self.x, self.y, None
        w = self.weight(self.x)
        return self.x, self.y, np.asarray(w, dtype=float)


@dataclass
class SimulationCase:
    template: FebTemplate
    subfolder: str
    experiments: Mapping[str, ExperimentSeries]
    adapters: Mapping[str, SimulationAdapter]
    tasks: TaskPipeline = field(default_factory=TaskPipeline)

    def __post_init__(self) -> None:
        missing = set(self.experiments.keys()) - set(self.adapters.keys())
        if missing:
            raise ValueError(f"Missing adapters for experiments: {sorted(missing)}")
        self._builder = FebBuilder(self.template, subfolder=self.subfolder)

    def prepare(
        self,
        theta: Mapping[str, float],
        out_root: Path,
        ctx: Optional[BuildContext] = None,
        context: Optional[TaskContext] = None,
    ) -> Tuple[Path, Path, TaskContext]:
        task_context = context or TaskContext()
        task_context.update({"theta": dict(theta)})
        task_context = self.tasks.run_pre(task_context)

        feb_path_str, xplt_str = self._builder.build(
            theta=dict(theta),
            out_root=str(out_root),
            ctx=ctx or BuildContext(iter_id=0, case_name=self.subfolder),
            out_name=f"{self.subfolder}.feb",
        )
        feb_path = Path(feb_path_str)
        xplt_path = Path(xplt_str)
        task_context["feb_path"] = feb_path
        task_context["xplt_path"] = xplt_path
        return feb_path, xplt_path, task_context

    def collect(
        self,
        xplt_path: Path,
        context: Optional[TaskContext] = None,
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], TaskContext]:
        task_context = context or TaskContext()
        results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, adapter in self.adapters.items():
            results[name] = adapter.read(xplt_path)
        task_context["results"] = results
        task_context = self.tasks.run_post(task_context)
        return results, task_context

