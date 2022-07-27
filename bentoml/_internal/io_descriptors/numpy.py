from __future__ import annotations

import json
import typing as t
import logging
from typing import overload
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from .json import MIME_TYPE_JSON
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import BentoMLException
from ...exceptions import InternalServerError
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody
from ...exceptions import UnprocessableEntity

if TYPE_CHECKING:
    import numpy as np

    from bentoml.grpc.v1 import service_pb2

    from .. import external_typing as ext
    from ..context import InferenceApiContext as Context
    from ..server.grpc.types import BentoServicerContext
else:
    np = LazyLoader("np", globals(), "numpy")
    service_pb2 = LazyLoader("service_pb2", globals(), "bentoml.grpc.v1.service_pb2")

logger = logging.getLogger(__name__)


_DTYPE_TO_FIELD_MAP = {
    "DT_BOOL": "bool_contents",
    "DT_FLOAT": "float_contents",
    "DT_COMPLEX64": "float_contents",
    "DT_STRING": "string_contents",
    "DT_DOUBLE": "double_contents",
    "DT_COMPLEX128": "double_contents",
    "DT_INT32": "int_contents",
    "DT_IN16": "int_contents",
    "DT_UINT16": "int_contents",
    "DT_INT8": "int_contents",
    "DT_UINT8": "int_contents",
    "DT_HALF": "int_contents",
    "DT_INT64": "long_contents",
    "DT_STRUCT": "struct_contents",
    "DT_UINT32": "uint32_contents",
    "DT_UINT64": "uint64_contents",
    # "DT_QINT32": "bytes_contents",
    # "DT_QINT16": "bytes_contents",
    # "DT_QUINT16": "bytes_contents",
    # "DT_QINT8": "bytes_contents",
    # "DT_QUINT8": "bytes_contents",
    # "DT_BFLOAT16": "int_contents",
}

_DTYPE_TO_STRING_MAP = {
    "DT_BOOL": "bool",
    "DT_FLOAT": "float32",
    "DT_COMPLEX64": "complex64",
    "DT_STRING": "<U",  # <U is little-edian unicode, S: zero-terminated bytes (not recommended)
    "DT_DOUBLE": "float64",
    "DT_COMPLEX128": "complex128",
    "DT_INT32": "int32",
    "DT_INT16": "int16",
    "DT_UINT16": "uint16",
    "DT_INT8": "int8",
    "DT_UINT8": "uint8",
    "DT_HALF": "float16",
    "DT_INT64": "int64",
    "DT_UINT32": "uint32",
    "DT_UINT64": "uint64",
}


# TODO: support DT_BFLOAT16, quantized data type
_NOT_SUPPORTED_DTYPE = [
    "DT_QINT32",
    "DT_QINT16",
    "DT_QUINT16",
    "DT_QINT8",
    "DT_QUINT8",
    "DT_BFLOAT16",
]


# Note that if DT_STRUCT is used, user need to specify np.dtype in NumpyNdarray
def get_dtype(
    datatype_string: str,
    *,
    struct_npdtype: np.dtype[t.Any] | None = None,
) -> np.dtype[t.Any] | None:
    if datatype_string == "DT_UNSPECIFIED":
        return
    elif datatype_string in _NOT_SUPPORTED_DTYPE:
        raise UnprocessableEntity(f"{datatype_string} is not yet supported.")
    elif datatype_string == "DT_STRUCT":
        assert (
            struct_npdtype
        ), "'dtype' is required in NumpyNdarray to use in conjunction with DT_STRUCT."
        return struct_npdtype
    else:
        return np.dtype(_DTYPE_TO_STRING_MAP[datatype_string])


@overload
def get_array_value(array: dict[str, str | bytes]) -> tuple[str, bytes, bool]:
    ...


@overload
def get_array_value(
    array: dict[str, str | list[t.Any]]
) -> tuple[str, list[t.Any], bool]:
    ...


# array_descriptor -> {"dtype": "DT_FLOAT", "float_contents": [1, 2, 3]}
def get_array_value(array: dict[str, t.Any]) -> tuple[str, list[t.Any] | bytes, bool]:
    # returns the array contents with whether the result is using bytes.
    dtype = t.cast(str, array.pop("dtype"))
    if _DTYPE_TO_FIELD_MAP[dtype] not in array:
        if "bytes_contents" not in array:
            raise BadInput(
                f"{dtype} requires specifying either '{_DTYPE_TO_FIELD_MAP[dtype]}' or 'bytes_contents' in the protobuf message."
            )
        content = array.pop("bytes_contents")
        assert isinstance(content, bytes)
        return dtype, content, True
    else:
        # all of the repeated fields can be represented as list.
        content = t.cast(t.List[t.Any], array.pop(_DTYPE_TO_FIELD_MAP[dtype]))
        return dtype, content, False


def _is_matched_shape(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    if (left is None) or (right is None):
        return False

    if len(left) != len(right):
        return False

    for i, j in zip(left, right):
        if i == -1 or j == -1:
            continue
        if i == j:
            continue
        return False
    return True


# TODO: when updating docs, add examples with gRPCurl
class NumpyNdarray(IODescriptor["ext.NpNDArray"]):
    """
    :obj:`NumpyNdarray` defines API specification for the inputs/outputs of a Service, where
    either inputs will be converted to or outputs will be converted from type
    :code:`numpy.ndarray` as specified in your API function signature.

    A sample service implementation:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       from typing import TYPE_CHECKING, Any

       import bentoml
       from bentoml.io import NumpyNdarray

       if TYPE_CHECKING:
           from numpy.typing import NDArray

       runner = bentoml.sklearn.get("sklearn_model_clf").to_runner()

       svc = bentoml.Service("iris-classifier", runners=[runner])

       @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
       def predict(input_arr: NDArray[Any]) -> NDArray[Any]:
           return runner.run(input_arr)

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

        % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

            .. code-block:: bash

                % curl -X POST -H "Content-Type: application/json" \\
                        --data '[[5,4,3,2]]' http://0.0.0.0:3000/predict

                # [1]%

        .. tab-item:: Python

            .. code-block:: python
               :caption: `request.py`

                import requests

                requests.post(
                    "http://0.0.0.0:3000/predict",
                    headers={"content-type": "application/json"},
                    data='[{"0":5,"1":4,"2":3,"3":2}]'
                ).text

    Args:
        dtype: Data type users wish to convert their inputs/outputs to. Refers to `arrays dtypes <https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_ for more information.
        enforce_dtype: Whether to enforce a certain data type. if :code:`enforce_dtype=True` then :code:`dtype` must be specified.
        shape: Given shape that an array will be converted to. For example:

               .. code-block:: python
                  :caption: `service.py`

                  from bentoml.io import NumpyNdarray

                  @svc.api(input=NumpyNdarray(shape=(2,2), enforce_shape=False), output=NumpyNdarray())
                  async def predict(input_array: np.ndarray) -> np.ndarray:
                      # input_array will be reshaped to (2,2)
                      result = await runner.run(input_array)

               When ``enforce_shape=True`` is provided, BentoML will raise an exception if the input array received does not match the `shape` provided.
        enforce_shape: Whether to enforce a certain shape. If ``enforce_shape=True`` then ``shape`` must be specified.

    Returns:
        :obj:`~bentoml._internal.io_descriptors.IODescriptor`: IO Descriptor that represents a :code:`np.ndarray`.
    """

    def __init__(
        self,
        dtype: str | ext.NpDTypeLike | None = None,
        enforce_dtype: bool = False,
        shape: tuple[int, ...] | None = None,
        enforce_shape: bool = False,
        packed: bool = False,
        bytesorder: t.Literal["C", "F", "A", None] = None,
    ):
        if dtype and not isinstance(dtype, np.dtype):
            # Convert from primitive type or type string, e.g.:
            # np.dtype(float)
            # np.dtype("float64")
            try:
                dtype = np.dtype(dtype)
            except TypeError as e:
                raise UnprocessableEntity(f'NumpyNdarray: Invalid dtype "{dtype}": {e}')

        self._dtype: ext.NpDTypeLike | None = dtype
        self._shape = shape
        self._enforce_dtype = enforce_dtype
        self._enforce_shape = enforce_shape
        self._packed = packed
        if bytesorder not in ["C", "F", "A", None]:
            raise BadInput(
                f"'bytesorder' must be one of ['C', 'F', 'A', 'None'], got {bytesorder} instead."
            )
        if not bytesorder:
            bytesorder = "C"  # default from numpy (C-order)
            # https://numpy.org/doc/stable/user/basics.byteswapping.html#introduction-to-byte-ordering-and-ndarrays
        self._bytesorder: t.Literal["C", "F", "A", None] = bytesorder

        self._sample_input = None

    def _openapi_types(self) -> str:
        # convert numpy dtypes to openapi compatible types.
        var_type = "integer"
        if self._dtype:
            name: str = self._dtype.name
            if name.startswith("float") or name.startswith("complex"):
                var_type = "number"
        return var_type

    def input_type(self) -> LazyType[ext.NpNDArray]:
        return LazyType("numpy", "ndarray")

    @property
    def sample_input(self) -> ext.NpNDArray | None:
        return self._sample_input

    @sample_input.setter
    def sample_input(self, value: ext.NpNDArray) -> None:
        self._sample_input = value

    def openapi_schema(self) -> Schema:
        # Note that we are yet provide
        # supports schemas for arrays that is > 2D.
        items = Schema(type=self._openapi_types())
        if self._shape and len(self._shape) > 1:
            items = Schema(type="array", items=Schema(type=self._openapi_types()))

        return Schema(type="array", items=items, nullable=True)

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self) -> t.Any:
        if self.sample_input is not None:
            if isinstance(self.sample_input, np.generic):
                raise BadInput("NumpyNdarray: sample_input must be a numpy array.")
            return self.sample_input.tolist()
        return

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            required=True,
        )

    def openapi_responses(self) -> OpenAPIResponse:
        return OpenAPIResponse(
            description=SUCCESS_DESCRIPTION,
            content={
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
        )

    def _verify_ndarray(
        self, obj: ext.NpNDArray, exception_cls: t.Type[Exception] = BadInput
    ) -> ext.NpNDArray:
        if self._dtype is not None and self._dtype != obj.dtype:
            # ‘same_kind’ means only safe casts or casts within a kind, like float64
            # to float32, are allowed.
            if np.can_cast(obj.dtype, self._dtype, casting="same_kind"):
                obj = obj.astype(self._dtype, casting="same_kind")  # type: ignore
            else:
                msg = f'{self.__class__.__name__}: Expecting ndarray of dtype "{self._dtype}", but "{obj.dtype}" was received.'
                if self._enforce_dtype:
                    raise exception_cls(msg)
                else:
                    logger.debug(msg)

        if self._shape is not None and not _is_matched_shape(self._shape, obj.shape):
            msg = f'{self.__class__.__name__}: Expecting ndarray of shape "{self._shape}", but "{obj.shape}" was received.'
            if self._enforce_shape:
                raise exception_cls(msg)
            try:
                obj = obj.reshape(self._shape)
            except ValueError as e:
                logger.debug(f"{msg} Failed to reshape: {e}.")

        return obj

    async def from_http_request(self, request: Request) -> ext.NpNDArray:
        """
        Process incoming requests and convert incoming objects to ``numpy.ndarray``.

        Args:
            request: Incoming Requests

        Returns:
            a ``numpy.ndarray`` object. This can then be used
             inside users defined logics.
        """
        obj = await request.json()
        try:
            res = np.array(obj, dtype=self._dtype)
        except ValueError:
            res = np.array(obj)
        return self._verify_ndarray(res)

    async def to_http_response(self, obj: ext.NpNDArray, ctx: Context | None = None):
        """
        Process given objects and convert it to HTTP response.

        Args:
            obj: ``np.ndarray`` that will be serialized to JSON
            ctx: ``Context`` object that contains information about the request.

        Returns:
            HTTP Response of type ``starlette.responses.Response``. This can
             be accessed via cURL or any external web traffic.
        """
        obj = self._verify_ndarray(obj, InternalServerError)
        if ctx is not None:
            res = Response(
                json.dumps(obj.tolist()),
                media_type=MIME_TYPE_JSON,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(json.dumps(obj.tolist()), media_type=MIME_TYPE_JSON)

    async def from_grpc_request(
        self, request: service_pb2.Request, context: BentoServicerContext
    ) -> ext.NpNDArray:
        """
        Process incoming protobuf request and convert it to `numpy.ndarray`

        Args:
            request: Incoming Requests

        Returns:
            a `numpy.ndarray` object. This can then be used
             inside users defined logics.
        """
        import grpc

        from ..utils.grpc import deserialize_proto

        field, serialized = deserialize_proto(self, request)
        if field == "ndarray_value":
            # {'shape': [2, 3], 'array': {'dtype': 'DT_FLOAT', ...}}
            if "array" not in serialized:
                msg = "'array' cannot be None."
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                raise BadInput(msg)

            shape = tuple(serialized["shape"])
            if self._shape:
                if not self._enforce_shape:
                    logger.warning(
                        f"'shape={self._shape},enforce_shape={self._enforce_shape}' is set with {self.__class__.__name__}, while 'shape' field is present in request message. To avoid this warning, set 'enforce_shape=True'. Using 'shape={shape}' from request message."
                    )
                    self._shape = shape
                else:
                    logger.debug(
                        f"'enforce_shape={self._enforce_shape}', ignoring 'shape' field in request message."
                    )
            else:
                self._shape = shape

            array = serialized["array"]
        else:
            # {'dtype': 'DT_FLOAT', 'float_contents': [1.0, 2.0, 3.0]}
            array = serialized

        dtype_string, content, use_bytes = get_array_value(array)
        dtype = get_dtype(dtype_string, struct_npdtype=self._dtype)
        if self._dtype:
            if not self._enforce_dtype:
                logger.warning(
                    f"'dtype={self._dtype},enforce_dtype={self._enforce_dtype}' is set with {self.__class__.__name__}, while 'dtype' field is present in request message. To avoid this warning, set 'enforce_dtype=True'. Using 'dtype={dtype}' from request message."
                )
                self._dtype = dtype
            else:
                logger.debug(
                    f"'enforce_dtype={self._enforce_dtype}', ignoring 'dtype' field in request message."
                )
        else:
            self._dtype = dtype

        if use_bytes:
            res = np.frombuffer(content, dtype=self._dtype)
        else:
            try:
                res = np.array(content, dtype=self._dtype)
            except ValueError:
                res = np.array(content)

        return self._verify_ndarray(res, BadInput)

    async def to_grpc_response(
        self, obj: ext.NpNDArray, context: BentoServicerContext
    ) -> service_pb2.Response:
        """
        Process given objects and convert it to grpc protobuf response.

        Args:
            obj: `np.ndarray` that will be serialized to protobuf
            context: grpc.aio.ServicerContext from grpc.aio.Server
        Returns:
            `io_descriptor_pb2.Array`:
                Protobuf representation of given `np.ndarray`
        """
        from ..utils.grpc import grpc_status_code
        from ..configuration import get_debug_mode

        _NPTYPE_TO_DTYPE_STRING_MAP = {
            np.dtype(v): k for k, v in _DTYPE_TO_STRING_MAP.items()
        }
        dtype_string = _NPTYPE_TO_DTYPE_STRING_MAP[obj.dtype]

        try:
            obj = self._verify_ndarray(obj, InternalServerError)
        except InternalServerError as e:
            context.set_code(grpc_status_code(e))
            context.set_details(e.message)
            raise

        cnt: dict[str, t.Any] = {"dtype": dtype_string}

        resp = service_pb2.Response()
        if self._packed:
            cnt.update({"bytes_contents": obj.tobytes(order=self._bytesorder)})
        else:
            if self._bytesorder:
                logger.warning(
                    f"'bytesorder={self._bytesorder}' is ignored when 'packed={self._packed}'."
                )
            cnt.update({_DTYPE_TO_FIELD_MAP[dtype_string]: obj.tolist()})

        if obj.ndim == 1:
            message = service_pb2.Array(**cnt)
            resp.contents.array_value.CopyFrom(message)
        else:
            cnt["shape"] = tuple(obj.shape)
            resp.contents.ndarray_value.CopyFrom(
                service_pb2.NDArray(
                    shape=tuple(obj.shape), array=service_pb2.Array(**cnt)
                )
            )
        if get_debug_mode():
            logger.debug(f"Response proto: \n{resp}")
        return resp

    def generate_protobuf(self):
        pass

    @classmethod
    def from_sample(
        cls,
        sample_input: ext.NpNDArray,
        enforce_dtype: bool = True,
        enforce_shape: bool = True,
    ) -> NumpyNdarray:
        """
        Create a :obj:`NumpyNdarray` IO Descriptor from given inputs.

        Args:
            sample_input: Given sample ``np.ndarray`` data
            enforce_dtype: Enforce a certain data type. :code:`dtype` must be specified at function
                           signature. If you don't want to enforce a specific dtype then change
                           :code:`enforce_dtype=False`.
            enforce_shape: Enforce a certain shape. :code:`shape` must be specified at function
                           signature. If you don't want to enforce a specific shape then change
                           :code:`enforce_shape=False`.

        Returns:
            :obj:`NumpyNdarray`: :code:`NumpyNdarray` IODescriptor from given users inputs.

        Example:

        .. code-block:: python
           :caption: `service.py`

           from __future__ import annotations

           from typing import TYPE_CHECKING, Any

           import bentoml
           from bentoml.io import NumpyNdarray

           import numpy as np

           if TYPE_CHECKING:
               from numpy.typing import NDArray

           input_spec = NumpyNdarray.from_sample(np.array([[1,2,3]]))

           @svc.api(input=input_spec, output=NumpyNdarray())
           async def predict(input: NDArray[np.int16]) -> NDArray[Any]:
               return await runner.async_run(input)
        """
        if isinstance(sample_input, np.generic):
            raise BentoMLException(
                "NumpyNdarray.from_sample() expects a numpy.array, not numpy.generic."
            )

        inst = cls(
            dtype=sample_input.dtype,
            shape=sample_input.shape,
            enforce_dtype=enforce_dtype,
            enforce_shape=enforce_shape,
        )
        inst.sample_input = sample_input

        return inst
