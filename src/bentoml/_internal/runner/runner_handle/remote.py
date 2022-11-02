from __future__ import annotations

import json
import pickle
import typing as t
import functools
from typing import TYPE_CHECKING
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

from bentoml._internal.utils.uri import uri_to_path

from . import RunnerHandle
from ...context import component_context
from ..container import Payload
from ....exceptions import RemoteException
from ....exceptions import ServiceUnavailable
from ...runner.utils import Params
from ...runner.utils import PAYLOAD_META_HEADER
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import httpx

    from ..runner import Runner
    from ..runner import RunnerMethod

    P = t.ParamSpec("P")
    R = t.TypeVar("R")


class RemoteRunnerClient(RunnerHandle):
    _conn: httpx.HTTPTransport | None = None
    _client_cache: httpx.Client | None = None
    _addr: str = ""

    def __init__(self, runner: Runner):  # pylint: disable=super-init-not-called
        self._runner = runner

    @property
    def _remote_runner_server_map(self) -> dict[str, str]:
        return BentoMLContainer.remote_runner_mapping.get()

    @property
    def runner_timeout(self) -> int:
        "return the configured timeout for this runner."
        runner_cfg = BentoMLContainer.runners_config.get()
        if self._runner.name in runner_cfg:
            return runner_cfg[self._runner.name]["timeout"]
        else:
            return runner_cfg["timeout"]

    def _close_conn(self) -> None:
        if self._conn:
            self._conn.close()

    def _get_conn(self) -> httpx.HTTPTransport:
        import httpx

        if self._conn is None:
            bind_uri = self._remote_runner_server_map[self._runner.name]
            parsed = urlparse(bind_uri)
            if parsed.scheme == "file":
                path = uri_to_path(bind_uri)
                self._conn = httpx.HTTPTransport(
                    uds=path,
                    limits=httpx.Limits(
                        max_connections=800,
                        max_keepalive_connections=800,
                        keepalive_expiry=1800,
                    ),  # TODO(jiang): make it configurable
                )
                self._addr = "http://0.0.0.0"
            elif parsed.scheme == "tcp":
                self._conn = httpx.HTTPTransport(
                    verify=False,
                    limits=httpx.Limits(
                        max_connections=800,
                        max_keepalive_connections=800,
                        keepalive_expiry=1800,
                    ),  # TODO(jiang): make it configurable
                )
                self._addr = f"http://{parsed.netloc}"
            else:
                raise ValueError(f"Unsupported bind scheme: {parsed.scheme}") from None
        return self._conn

    @property
    def _client(
        self,
    ) -> httpx.Client:
        import httpx

        if self._client_cache is None or self._client_cache.is_closed:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            self._client_cache = httpx.Client(
                transport=self._get_conn(),
                base_url=self._addr,
                timeout=self.runner_timeout,
                trust_env=True,
            )
            HTTPXClientInstrumentor.instrument_client(self._client_cache)
        return self._client_cache

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | tuple[R, ...]:
        from ...runner.container import AutoContainer

        inp_batch_dim = __bentoml_method.config.batch_dim[0]

        payload_params = Params[Payload](*args, **kwargs).map(
            functools.partial(AutoContainer.to_payload, batch_dim=inp_batch_dim)
        )

        if __bentoml_method.config.batchable:
            if not payload_params.map(lambda i: i.batch_size).all_equal():
                raise ValueError(
                    "All batchable arguments must have the same batch size."
                ) from None

        path = "" if __bentoml_method.name == "__call__" else __bentoml_method.name
        try:
            resp: httpx.Response = self._client.post(
                url=f"/{path}",
                content=pickle.dumps(payload_params),  # FIXME: pickle inside pickle
                headers={
                    "Bento-Name": component_context.bento_name,
                    "Bento-Version": component_context.bento_version,
                    "Runner-Name": self._runner.name,
                    "Yatai-Bento-Deployment-Name": component_context.yatai_bento_deployment_name,
                    "Yatai-Bento-Deployment-Namespace": component_context.yatai_bento_deployment_namespace,
                },
            )
            body = resp.read()
        except Exception as e:
            raise RemoteException(f"Failed to connect to runner server.") from e

        try:
            content_type = resp.headers["Content-Type"]
            assert content_type.lower().startswith("application/vnd.bentoml.")
        except (KeyError, AssertionError):
            raise RemoteException(
                f"An unexpected exception occurred in remote runner {self._runner.name}: [{resp.status_code}] {body.decode()}"
            ) from None

        if resp.status_code != 200:
            if resp.status_code == 503:
                raise ServiceUnavailable(body.decode()) from None
            if resp.status_code == 500:
                raise RemoteException(body.decode()) from None
            raise RemoteException(
                f"An exception occurred in remote runner {self._runner.name}: [{resp.status_code}] {body.decode()}"
            ) from None

        try:
            meta_header = resp.headers[PAYLOAD_META_HEADER]
        except KeyError:
            raise RemoteException(
                f"Bento payload decode error: {PAYLOAD_META_HEADER} header not set. An exception might have occurred in the remote server. [{resp.status_code}] {body.decode()}"
            ) from None

        if content_type == "application/vnd.bentoml.multiple_outputs":
            payloads = pickle.loads(body)
            return tuple(AutoContainer.from_payload(payload) for payload in payloads)

        container = content_type.strip("application/vnd.bentoml.")

        try:
            payload = Payload(
                data=body, meta=json.loads(meta_header), container=container
            )
        except JSONDecodeError:
            raise ValueError(f"Bento payload decode error: {meta_header}") from None

        return AutoContainer.from_payload(payload)

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        import anyio

        return t.cast(
            "R",
            anyio.from_thread.run(
                self.async_run_method,
                __bentoml_method,
                *args,
                **kwargs,
            ),
        )

    async def is_ready(self, timeout: int) -> bool:
        # default kubernetes probe timeout is also 1s; see
        # https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#configure-probes
        resp = self._client.get(
            "/readyz",
            headers={
                "Bento-Name": component_context.bento_name,
                "Bento-Version": component_context.bento_version,
                "Runner-Name": self._runner.name,
                "Yatai-Bento-Deployment-Name": component_context.yatai_bento_deployment_name,
                "Yatai-Bento-Deployment-Namespace": component_context.yatai_bento_deployment_namespace,
            },
            timeout=timeout,
        )
        return resp.status_code == 200

    def __del__(self) -> None:
        self._close_conn()
