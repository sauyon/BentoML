"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import bentoml.grpc.v1.service_pb2
import grpc
import typing

class BentoServiceStub:
    """a gRPC BentoServer."""
    def __init__(self, channel: grpc.Channel) -> None: ...
    ServerLive: grpc.UnaryUnaryMultiCallable[
        bentoml.grpc.v1.service_pb2.ServerLiveRequest,
        bentoml.grpc.v1.service_pb2.ServerLiveResponse]
    """Check server liveliness."""

    ServerReady: grpc.UnaryUnaryMultiCallable[
        bentoml.grpc.v1.service_pb2.ServerReadyRequest,
        bentoml.grpc.v1.service_pb2.ServerReadyResponse]
    """Check server readiness"""

    Call: grpc.UnaryUnaryMultiCallable[
        bentoml.grpc.v1.service_pb2.CallRequest,
        bentoml.grpc.v1.service_pb2.CallResponse]
    """Call handles unary API."""

    CallStream: grpc.StreamStreamMultiCallable[
        bentoml.grpc.v1.service_pb2.CallStreamRequest,
        bentoml.grpc.v1.service_pb2.CallStreamResponse]
    """CallStream handles streaming API."""


class BentoServiceServicer(metaclass=abc.ABCMeta):
    """a gRPC BentoServer."""
    @abc.abstractmethod
    def ServerLive(self,
        request: bentoml.grpc.v1.service_pb2.ServerLiveRequest,
        context: grpc.ServicerContext,
    ) -> bentoml.grpc.v1.service_pb2.ServerLiveResponse:
        """Check server liveliness."""
        pass

    @abc.abstractmethod
    def ServerReady(self,
        request: bentoml.grpc.v1.service_pb2.ServerReadyRequest,
        context: grpc.ServicerContext,
    ) -> bentoml.grpc.v1.service_pb2.ServerReadyResponse:
        """Check server readiness"""
        pass

    @abc.abstractmethod
    def Call(self,
        request: bentoml.grpc.v1.service_pb2.CallRequest,
        context: grpc.ServicerContext,
    ) -> bentoml.grpc.v1.service_pb2.CallResponse:
        """Call handles unary API."""
        pass

    @abc.abstractmethod
    def CallStream(self,
        request_iterator: typing.Iterator[bentoml.grpc.v1.service_pb2.CallStreamRequest],
        context: grpc.ServicerContext,
    ) -> typing.Iterator[bentoml.grpc.v1.service_pb2.CallStreamResponse]:
        """CallStream handles streaming API."""
        pass


def add_BentoServiceServicer_to_server(servicer: BentoServiceServicer, server: grpc.Server) -> None: ...
