from __future__ import annotations

import io
import sys
import json
import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pytest

from bentoml.testing.utils import async_request
from bentoml.testing.utils import parse_multipart_form

if TYPE_CHECKING:
    import PIL.Image as PILImage
else:
    from bentoml._internal.utils import LazyLoader

    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")


@pytest.mark.asyncio
async def test_numpy(host: str):
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_shape",
        headers={"Content-Type": "application/json"},
        content="[[1,2],[3,4]]",
        assert_status=200,
        assert_data=b"[[2, 4], [6, 8]]",
    )
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_multi_output",
        headers={"Content-Type": "application/json"},
        content="[[1,2],[3,4]]",
        assert_status=200,
        assert_data=b"[[2, 4], [6, 8]]",
    )
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_shape",
        headers={"Content-Type": "application/json"},
        content="[1,2,3,4]",
        assert_status=400,
    )
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_dtype",
        headers={"Content-Type": "application/json"},
        content="[[2,1],[4,3]]",
        assert_status=200,
        assert_data=b'[["4", "2"], ["8", "6"]]',
    )
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_dtype",
        headers={"Content-Type": "application/json"},
        content='[["2f",1],[4,3]]',
        assert_status=400,
    )


@pytest.mark.asyncio
async def test_json(host: str):
    ORIGIN = "http://bentoml.ai"

    await async_request(
        "POST",
        f"http://{host}/echo_json",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        content='"hi"',
        assert_status=200,
        assert_data=b'"hi"',
    )

    await async_request(
        "POST",
        f"http://{host}/echo_json_sync",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        content='"hi"',
        assert_status=200,
        assert_data=b'"hi"',
    )

    await async_request(
        "POST",
        f"http://{host}/echo_json_enforce_structure",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        content='{"name":"test","endpoints":["predict","health"]}',
        assert_status=200,
        assert_data=b'{"name":"test","endpoints":["predict","health"]}',
    )


@pytest.mark.asyncio
async def test_obj(host: str):
    for obj in [1, 2.2, "str", [1, 2, 3], {"a": 1, "b": 2}]:
        obj_str = json.dumps(obj, separators=(",", ":"))
        await async_request(
            "POST",
            f"http://{host}/echo_obj",
            headers=(("Content-Type", "application/json"),),
            content=obj_str,
            assert_status=200,
            assert_data=obj_str.encode("utf-8"),
        )


@pytest.mark.asyncio
async def test_pandas(host: str):
    import pandas as pd

    ORIGIN = "http://bentoml.ai"

    df = pd.DataFrame([[101]], columns=["col1"])

    await async_request(
        "POST",
        f"http://{host}/predict_dataframe",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        content=df.to_json(orient="records"),
        assert_status=200,
        assert_data=b'[{"col1":202}]',
    )

    # pyarrow only support python 3.7+
    if sys.version_info >= (3, 7):
        await async_request(
            "POST",
            f"http://{host}/predict_dataframe",
            headers=(("Content-Type", "application/octet-stream"), ("Origin", ORIGIN)),
            content=df.to_parquet(),
            assert_status=200,
            assert_data=b'[{"col1":202}]',
        )

    await async_request(
        "POST",
        f"http://{host}/predict_dataframe",
        headers=(("Content-Type", "text/csv"), ("Origin", ORIGIN)),
        content=df.to_csv(),
        assert_status=200,
        assert_data=b'[{"col1":202}]',
    )


@pytest.mark.asyncio
async def test_file(host: str, bin_file: str):
    # Test File as binary
    with open(str(bin_file), "rb") as f:
        b = f.read()

    await async_request(
        "POST",
        f"http://{host}/predict_file",
        content=b,
        headers={"Content-Type": "application/octet-stream"},
        assert_data=b"\x810\x899",
    )

    # Test File as multipart binary
    await async_request(
        "POST",
        f"http://{host}/predict_file",
        files={"file": ("file", b, "application/octet-stream")},
        assert_data=b"\x810\x899",
    )

    # Test Exception
    await async_request(
        "POST",
        f"http://{host}/predict_file",
        content=b,
        headers={"Content-Type": "application/pdf"},
        assert_status=500,
    )


@pytest.mark.asyncio
async def test_image(host: str, img_file: str):
    with open(str(img_file), "rb") as f1:
        img_bytes = f1.read()

    status, headers, body = await async_request(
        "POST",
        f"http://{host}/echo_image",
        content=img_bytes,
        headers={"Content-Type": "image/bmp"},
    )
    assert status == 200
    assert headers["Content-Type"] == "image/bmp"

    bio = io.BytesIO(body)
    bio.name = "test.bmp"
    img = PILImage.open(bio)
    array1 = np.array(img)
    array2 = PILImage.open(img_file)

    np.testing.assert_array_almost_equal(array1, np.array(array2))

    await async_request(
        "POST",
        f"http://{host}/echo_image",
        content=img_bytes,
        headers={"Content-Type": "application/json"},
        assert_status=400,
    )

    with open(str(img_file), "rb") as f1:
        b = f1.read()
    await async_request(
        "POST",
        f"http://{host}/echo_image",
        content=b,
        headers={"Content-Type": "application/pdf"},
        assert_status=400,
    )


@pytest.fixture(name="img_form_data")
def fixture_img_form_data(img_file: str) -> dict[str, tuple[str, t.BinaryIO, str]]:
    with open(img_file, "rb") as f1, open(img_file, "rb") as f2:
        return {
            "original": ("original", f1, "image/bmp"),
            "compared": ("compared", f2, "image/bmp"),
        }


@pytest.mark.asyncio
async def test_multipart_image_io(
    host: str, img_form_data: dict[str, tuple[str, t.BinaryIO, str]]
):
    from starlette.datastructures import UploadFile

    _, headers, body = await async_request(
        "POST",
        f"http://{host}/predict_multi_images",
        files=img_form_data,
        assert_status=200,
    )

    form = await parse_multipart_form(headers=headers, body=body)
    for _, v in form.items():
        assert isinstance(v, UploadFile)
        img = PILImage.open(
            v.file  # type: ignore # UploadFile has no type
        )
        assert np.array(img).shape == (10, 10, 3)


@pytest.mark.asyncio
async def test_multipart_image_io_1(host: str, img_form_data: dict[str, t.Any]):
    await async_request(
        "POST",
        f"http://{host}/predict_different_args",
        files=img_form_data,
        assert_status=200,
    )
