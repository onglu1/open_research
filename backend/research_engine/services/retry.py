"""重试辅助工具。"""

from __future__ import annotations


DEFAULT_RETRY_WAIT_SECONDS = 10


def get_retry_wait_seconds(retry_index: int, *, base_seconds: int = DEFAULT_RETRY_WAIT_SECONDS) -> int:
    """
    返回第 N 次重试前的等待时间。

    retry_index 从 1 开始：
    - 第 1 次重试: 10s
    - 第 2 次重试: 20s
    - 第 3 次重试: 40s
    """
    if retry_index < 1:
        raise ValueError("retry_index 必须从 1 开始")
    return base_seconds * (2 ** (retry_index - 1))
