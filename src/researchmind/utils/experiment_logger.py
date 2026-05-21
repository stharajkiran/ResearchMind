from abc import ABC, abstractmethod
from typing import Any


class ExperimentLogger(ABC):
    """Wrapper around experiment tracking tools.

    Implement this to swap between MLflow, W&B, Neptune, or a no-op
    without touching any evaluation or training code.

    Usage (same pattern for all implementations):
        with logger.start_run("my_run", experiment_name="MyExperiment"):
            logger.log_params({"lr": 0.001})
            logger.log_metric("recall", 0.87)
    """

    @abstractmethod
    def start_run(
        self, run_name: str, experiment_name: str | None = None
    ) -> "ExperimentLogger":
        """Start a tracking run. Returns self so it can be used as a context manager."""
        ...

    @abstractmethod
    def end_run(self) -> None:
        """End the current run."""
        ...

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.end_run()

    @property
    @abstractmethod
    def run_id(self) -> str | None:
        """Active run ID, or None if no run is active."""
        ...

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None: ...

    @abstractmethod
    def log_dict(self, data: dict, filename: str) -> None: ...

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None: ...


class MLflowLogger(ExperimentLogger):
    """ExperimentLogger backed by MLflow."""

    def start_run(
        self, run_name: str, experiment_name: str | None = None
    ) -> "MLflowLogger":
        import mlflow
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        return self

    def end_run(self) -> None:
        import mlflow
        mlflow.end_run()

    @property
    def run_id(self) -> str | None:
        import mlflow
        active = mlflow.active_run()
        return active.info.run_id if active else None

    def log_param(self, key: str, value: Any) -> None:
        import mlflow
        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        import mlflow
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        import mlflow
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        import mlflow
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dict(self, data: dict, filename: str) -> None:
        import mlflow
        mlflow.log_dict(data, filename)

    def set_tag(self, key: str, value: str) -> None:
        import mlflow
        mlflow.set_tag(key, value)


class NoOpLogger(ExperimentLogger):
    """Drop-in ExperimentLogger that does nothing — for tests and CI runs."""

    def start_run(
        self, run_name: str, experiment_name: str | None = None
    ) -> "NoOpLogger":
        return self

    def end_run(self) -> None:
        pass

    @property
    def run_id(self) -> str | None:
        return None

    def log_param(self, key: str, value: Any) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        pass

    def log_dict(self, data: dict, filename: str) -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        pass
