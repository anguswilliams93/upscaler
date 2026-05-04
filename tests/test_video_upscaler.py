"""Pipeline tests with mocked ffmpeg + GPU."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from video_upscaler import VideoMeta


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_chunked_inference_calls_empty_cache(tmp_path) -> None:
    """Chunk loop runs ceil(N/CHUNK_FRAMES) inference passes and clears cache each."""
    import video_upscaler as vu

    fake_frames = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(12)]
    fake_meta = VideoMeta(
        width=64, height=64, fps=30.0, duration_s=0.4,
        total_frames=12, has_audio=False,
    )

    fake_pipe = MagicMock()
    fake_pipe.upscale_video_file.side_effect = NotImplementedError
    fake_pipe.upscale_chunk.side_effect = lambda t: torch.zeros(
        t.shape[0], 3, t.shape[2] * 4, t.shape[3] * 4,
        dtype=torch.float16, device=t.device,
    )

    out = tmp_path / "out.mp4"
    tmp_video = str(out) + ".noaudio.mp4"
    writer = MagicMock()
    writer.close.side_effect = lambda: open(tmp_video, "wb").close()

    with patch.object(vu, "probe", return_value=fake_meta), \
         patch.object(vu, "get_pipeline", return_value=fake_pipe), \
         patch.object(vu, "_decode_frames", return_value=iter(fake_frames)), \
         patch.object(vu, "_FrameWriter", return_value=writer), \
         patch.object(torch.cuda, "empty_cache") as ec:
        vu.upscale_video(
            input_path="ignored.mp4",
            output_path=str(out),
            model_key="seedvr2-7b",
            models_dir=str(tmp_path),
        )

    assert fake_pipe.upscale_chunk.call_count == 3  # ceil(12/5)
    assert ec.call_count >= 3
    assert writer.write.call_count == 12
    writer.close.assert_called_once()
