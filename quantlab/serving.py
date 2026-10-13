import asyncio
import torch
from contextlib import AbstractContextManager
import logging
import os


class MlflowTracker(AbstractContextManager):
    """Single-threaded optional tracking; never suppress training errors."""
    def __init__(self, experiment, run_name, params, tags, backend=None):
        self.experiment, self.run_name = experiment, run_name
        self.params, self.tags = params, tags
        self.failed, self._active = False, False
        self._backend = backend
        if backend is None and os.environ.get("MLFLOW_TRACKING_URI", "").strip():
            try:
                import mlflow
                self._backend = mlflow
            except ImportError:
                self.failed = True
                logging.warning("MLflow unavailable; tracking disabled")

    def _call(self, method, *args, **kwargs):
        if self._backend is None or self.failed:
            return None
        try:
            return getattr(self._backend, method)(*args, **kwargs)
        except Exception:
            self.failed = True
            logging.exception("Tracking failed; retain local training results")
            return None

    def __enter__(self):
        self._call("set_experiment", self.experiment)
        run = self._call("start_run", run_name=self.run_name)
        self._active = run is not None
        self._call("log_params", self.params)
        self._call("set_tags", self.tags)
        return self

    def log_metrics(self, metrics, step):
        self._call("log_metrics", metrics, step=step)

    def log_artifact(self, path):
        self._call("log_artifact", path)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._active:
            try:
                status = "FAILED" if exc_type or self.failed else "FINISHED"
                self._backend.end_run(status=status)
            except Exception:
                self.failed = True
                logging.exception("Could not close tracking run")
            finally:
                self._active = False
        return False



class SACServingState:
    """CPU deterministic-action adapter; loader validates model/schema metadata."""
    def __init__(self, checkpoint_path, loader, state_dim=11):
        self.state_dim = state_dim
        self.loader = loader
        self.lock = asyncio.Lock()
        self.reload_lock = asyncio.Lock()
        self.actor = self._prepare(checkpoint_path)
        self.checkpoint_path = checkpoint_path

    def _infer(self, actor, state):
        tensor = torch.as_tensor(state, dtype=torch.float32)
        if tensor.shape != (self.state_dim,) or not torch.isfinite(tensor).all():
            raise ValueError("finite state with saved feature dimension required")
        with torch.inference_mode():
            output = actor(tensor.unsqueeze(0))
        if not isinstance(output, torch.Tensor) or output.shape != (1, 1):
            raise ValueError("adapter must return one deterministic action")
        value = float(output.item())
        if not torch.isfinite(output).all() or not -1 <= value <= 1:
            raise ValueError("finite action in [-1, 1] required")
        return value

    def _prepare(self, path):
        candidate = self.loader(path).to("cpu").eval()
        self._infer(candidate, [0.0] * self.state_dim)
        return candidate

    async def reload(self, checkpoint_path):
        async with self.reload_lock:
            candidate = await asyncio.to_thread(self._prepare, checkpoint_path)
            async with self.lock:
                self.actor = candidate
                self.checkpoint_path = checkpoint_path

    async def predict(self, state: list[float]) -> int:
        async with self.lock:
            actor = self.actor
        value = await asyncio.to_thread(self._infer, actor, state)
        return 1 if value > 0.3 else (-1 if value < -0.3 else 0)


def train_and_track(returns, tracking_backend=None, steps=128):
    """A complete SAC training/logging exercise using an 11D toy environment."""
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from stable_baselines3 import SAC
    from tiny_env import TinyTradingEnv
    env = TinyTradingEnv(returns)
    agent = SAC("MlpPolicy", env, learning_starts=32, buffer_size=512,
                batch_size=16, policy_kwargs={"net_arch":[16,16]}, device="cpu", seed=7)
    with TemporaryDirectory() as directory:
        with MlflowTracker("book-sac", "local-exercise", {"steps": steps},
                           {"mode":"learning"}, backend=tracking_backend) as tracker:
            agent.learn(total_timesteps=steps)
            tracker.log_metrics({"training_steps":float(agent.num_timesteps)}, step=0)
            path = Path(directory) / "sac.zip"
            agent.save(path)
            tracker.log_artifact(str(path))
    return agent


def predict_actor(actor, state, state_dim=11):
    tensor = torch.as_tensor(state, dtype=torch.float32)
    if tensor.shape != (state_dim,) or not torch.isfinite(tensor).all():
        raise ValueError("finite state matching the saved schema required")
    actor = actor.to("cpu").eval()
    with torch.inference_mode():
        return actor(tensor.unsqueeze(0))


def compile_jit(model):
    """Compile a compatible torch module, not an arbitrary SAC checkpoint."""
    import copy
    scripted = torch.jit.script(copy.deepcopy(model).eval())
    return torch.jit.freeze(scripted)


async def download_and_reload(run_id, artifact_path, serving, downloader):
    """Use an injected artifact downloader, then the validated local reload API."""
    if not run_id or not artifact_path:
        raise ValueError("run ID and artifact path required")
    path = await asyncio.to_thread(downloader, run_id=run_id, artifact_path=artifact_path)
    await serving.reload(path)
    return serving.checkpoint_path
