import argparse
import asyncio
import io
import json
import logging
import os
import re
import ssl
import time
import uuid

import numpy as np
import soundfile
import torch
import torchaudio
import websockets
from speechbrain.pretrained import SpeakerRecognition

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("asr_log.log")
file_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

websocket_users = set()

logger.info("model loading")

# 加载模型
spkrec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

logger.info("model loaded! only support one client at the same time now!!!!")


def bytes_to_tensor(pcm_bytes: bytes, bit_depth: int = 16) -> torch.Tensor:
    # 将字节流转换为numpy数组
    dtype = np.int16 if bit_depth == 16 else np.int32
    np_audio = np.frombuffer(pcm_bytes, dtype=dtype)

    # 转Tensor并归一化
    tensor = torch.from_numpy(np_audio.astype(np.float32))
    if bit_depth == 16:
        tensor /= 32768.0  # int16范围归一化到[-1, 1]
    elif bit_depth == 32:
        tensor /= 2147483648.0  # int32范围归一化
    return tensor.unsqueeze(0)  # 添加通道维度 (1, N)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """计算余弦相似度"""
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).item()


if __name__ == "__main__":
    with open("qiao.mp3", "rb") as f:
        audio_a = f.read()

    with open("guo.mp3", "rb") as f:
        audio_b = f.read()
    signal_a = bytes_to_tensor(audio_a)
    signal_b = bytes_to_tensor(audio_b)
    audio_emb_a = spkrec.encode_batch(signal_a).squeeze(0)
    audio_emb_b = spkrec.encode_batch(signal_b).squeeze(0)
    sim = cosine_similarity(audio_emb_a, audio_emb_b)
    logger.info(f"cosine_similarity:  {sim}")
