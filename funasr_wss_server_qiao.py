import argparse
import asyncio
import io
import json
import logging
import os
import ssl
import time
import uuid

import numpy as np
import soundfile
import torch
import torchaudio
import websockets
from speechbrain.pretrained import SpeakerRecognition

IS_SPEAKER_VERIFICATION = os.getenv("IS_SPEAKER_VERIFICATION", True)

SPEAKER_VERIFICATION_THRESHOLD = os.getenv("SPEAKER_VERIFICATION_THRESHOLD", 0.2)

SPEAKER_VERIFICATION_CHUNK_DURATION = os.getenv("SPEAKER_VERIFICATION_CHUNK_DURATION", 2.0)

HXQ_ROLE_KEYWORDS = ["心心", "欣欣", "星星"]

DEFAULT_SAMPLE_RATE = 16000


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('asr_log.log')
file_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    required=False,
    help="host ip, localhost, 0.0.0.0",
)
parser.add_argument(
    "--port", type=int, default=10095, required=False, help="grpc server port"
)
parser.add_argument(
    "--asr_model",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    help="model from modelscope",
)
parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--asr_model_online",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    help="model from modelscope",
)
parser.add_argument("--asr_model_online_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    help="model from modelscope",
)
parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--punc_model",
    type=str,
    default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    help="model from modelscope",
)
parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
parser.add_argument(
    "--certfile",
    type=str,
    default="runtime/ssl_key/server.crt",
    required=False,
    help="certfile for ssl",
)

parser.add_argument(
    "--keyfile",
    type=str,
    default="runtime/ssl_key/server.key",
    required=False,
    help="keyfile for ssl",
)
args = parser.parse_args()


websocket_users = set()

logger.info("model loading")
from funasr import AutoModel

# asr
model_asr = AutoModel(
    model=args.asr_model,
    model_revision=args.asr_model_revision,
    spk_model="cam++",
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)
# asr
model_asr_streaming = AutoModel(
    model=args.asr_model_online,
    model_revision=args.asr_model_online_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)
# vad
model_vad = AutoModel(
    model=args.vad_model,
    model_revision=args.vad_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    # chunk_size=60,
)

if args.punc_model != "":
    model_punc = AutoModel(
        model=args.punc_model,
        model_revision=args.punc_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )
else:
    model_punc = None


# 加载模型
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec"
)

logger.info("model loaded! only support one client at the same time now!!!!")


# sample_audio = "qiao_16k.wav"
# waveform, sample_rate = torchaudio.load(sample_audio)
# default_sample_emb = spkrec.encode_batch(waveform).squeeze(0)


def load_audio_from_bytes(audio_bytes):
    byte_io = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(byte_io)

    return waveform, sample_rate


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


async def ws_reset(websocket):
    logger.info(f"ws reset now, total num is len(websocket_users)")

    websocket.status_dict_asr_online["cache"] = {}
    websocket.status_dict_asr_online["is_final"] = True
    websocket.status_dict_vad["cache"] = {}
    websocket.status_dict_vad["is_final"] = True
    websocket.status_dict_punc["cache"] = {}

    await websocket.close()


async def clear_websocket():
    for websocket in websocket_users:
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path):
    frames = []
    frames_asr = []
    frames_asr_online = []
    global websocket_users
    # await clear_websocket()
    websocket_users.add(websocket)
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}
    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1
    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    websocket.speaker_verification_activate = False
    websocket.speaker_verification_sample_emb = None
    logger.info(f"new user connected: websocket:{websocket}")

    try:
        async for message in websocket:
            if isinstance(message, str):
                messagejson = json.loads(message)
                # logger.info(f"messagejson: {messagejson}")

                if "is_speaking" in messagejson:
                    websocket.is_speaking = messagejson["is_speaking"]
                    websocket.status_dict_asr_online[
                        "is_final"
                    ] = not websocket.is_speaking
                if "chunk_interval" in messagejson:
                    websocket.chunk_interval = messagejson["chunk_interval"]
                if "wav_name" in messagejson:
                    websocket.wav_name = messagejson.get("wav_name")
                if "chunk_size" in messagejson:
                    chunk_size = messagejson["chunk_size"]
                    if isinstance(chunk_size, str):
                        chunk_size = chunk_size.split(",")
                    websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
                if "encoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online[
                        "encoder_chunk_look_back"
                    ] = messagejson["encoder_chunk_look_back"]
                if "decoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online[
                        "decoder_chunk_look_back"
                    ] = messagejson["decoder_chunk_look_back"]
                if "hotwords" in messagejson:
                    websocket.status_dict_asr["hotword"] = messagejson["hotwords"]
                if "mode" in messagejson:
                    websocket.mode = messagejson["mode"]

            websocket.status_dict_vad["chunk_size"] = int(
                websocket.status_dict_asr_online["chunk_size"][1]
                * 60
                / websocket.chunk_interval
            )
            if (
                len(frames_asr_online) > 0
                or len(frames_asr) >= 0
                or not isinstance(message, str)
            ):
                if not isinstance(message, str):
                    frames.append(message)
                    duration_ms = len(message) // 32
                    websocket.vad_pre_idx += duration_ms

                    # asr online
                    frames_asr_online.append(message)
                    websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                    if (
                        len(frames_asr_online) % websocket.chunk_interval == 0
                        or websocket.status_dict_asr_online["is_final"]
                    ):
                        if websocket.mode == "2pass" or websocket.mode == "online":
                            audio_in = b"".join(frames_asr_online)
                            try:
                                await async_asr_online(websocket, audio_in)
                            except Exception as e:
                                logger.error(
                                    f"error in asr streaming, error: {e}"
                                )
                        frames_asr_online = []
                    if speech_start:
                        frames_asr.append(message)
                    # vad online
                    try:
                        speech_start_i, speech_end_i = await async_vad(
                            websocket, message
                        )
                    except:
                        logger.error("error in vad")
                    if speech_start_i != -1:
                        speech_start = True
                        beg_bias = (
                            websocket.vad_pre_idx - speech_start_i
                        ) // duration_ms
                        frames_pre = frames[-beg_bias:]
                        frames_asr = []
                        frames_asr.extend(frames_pre)
                # asr punc offline
                if speech_end_i != -1 or not websocket.is_speaking:
                    if websocket.mode == "2pass" or websocket.mode == "offline":
                        audio_in = b"".join(frames_asr)
                        try:
                            await async_asr(websocket, audio_in)
                        except Exception as e:
                            raise
                            logger.error(f"error in asr offline: error: {e}")
                    frames_asr = []
                    speech_start = False
                    frames_asr_online = []
                    websocket.status_dict_asr_online["cache"] = {}
                    if not websocket.is_speaking:
                        websocket.vad_pre_idx = 0
                        frames = []
                        websocket.status_dict_vad["cache"] = {}
                    else:
                        frames = frames[-20:]

    except:
        raise


async def async_vad(websocket, audio_in):
    segments_result = model_vad.generate(input=audio_in, **websocket.status_dict_vad)[
        0
    ]["value"]

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


def speaker_verify(websocket, audio_in, text):
    signal = bytes_to_tensor(audio_in)
    audio_emb = spkrec.encode_batch(signal).squeeze(0)
    if websocket.speaker_verification_activate:
        score = torch.nn.functional.cosine_similarity(
            websocket.speaker_verification_sample_emb, audio_emb, dim=1
        )
        logger.info(f"async_asr score : {score.item()}")
        if score.item() < SPEAKER_VERIFICATION_THRESHOLD:
            logger.info(f"no sample user audio!")
            return False
        else:
            return True
    else:
        if text and any(keyword in text for keyword in HXQ_ROLE_KEYWORDS):
            logger.info(f"keyword audio_emb")
            websocket.speaker_verification_sample_emb = audio_emb
            websocket.speaker_verification_activate = True
            return True
        else:
            return False


def extract_embedding(audio: torch.Tensor) -> torch.Tensor:
    """提取单段音频的声纹嵌入"""
    return spkrec.encode_batch(audio.squeeze(0))  # 输入形状: (1, samples)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """计算余弦相似度"""
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).item()


def process_audio_stream(audio_stream: np.ndarray, target_embed: torch.Tensor) -> torch.Tensor:
    """处理单个音频块并返回静音过滤后的结果"""
    buffer = torch.zeros(1, 0)  # 音频缓冲池
    # 转换为Tensor并缓冲
    chunk_tensor = torch.from_numpy(audio_stream).float().unsqueeze(0)
    buffer = torch.cat([buffer, chunk_tensor], dim=1)
    output = np.zeros_like(audio_stream)
    chunk_size = int(SPEAKER_VERIFICATION_CHUNK_DURATION * DEFAULT_SAMPLE_RATE)
    has_similarity = False
    while buffer.shape[1] >= chunk_size:
        # 提取待处理段
        segment = buffer[:, :chunk_size]

        # 声纹比对
        seg_embed = extract_embedding(segment)
        sim = cosine_similarity(target_embed, seg_embed)

        logger.info(f"process_audio_stream sim:  {sim}")

        # 静音替换逻辑
        if sim >= SPEAKER_VERIFICATION_THRESHOLD:
            output[:chunk_size] = segment
            has_similarity = True

        # 滑动窗口（50%重叠）
        buffer = buffer[:, chunk_size // 2:]

    return output, has_similarity


async def async_asr(websocket, audio_in):
    if len(audio_in) > 0:
        start_time = time.time()
        rec_result = model_asr.generate(input=audio_in, **websocket.status_dict_asr)[0]
        logger.info(f"offline_asr rec_result:  {rec_result}")
        text = rec_result["text"]
        if IS_SPEAKER_VERIFICATION:
            if websocket.speaker_verification_activate:
                chunk_np = np.frombuffer(audio_in, dtype=np.int16).astype(np.float32) / 32768.0
                processed, has_similarity = process_audio_stream(chunk_np, websocket.speaker_verification_sample_emb)
                if has_similarity:
                    rec_result = model_asr.generate(input=processed, **websocket.status_dict_asr)[0]
                    logger.info(f"offline_asr process_audio_stream   rec_result: {rec_result}")
                else:
                    mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
                    message = json.dumps(
                        {
                            "mode": mode,
                            "text": "",
                            "wav_name": websocket.wav_name,
                            "is_final": websocket.is_speaking,
                        }
                    )
                    await websocket.send(message)
                    return
            else:
                if text and any(keyword in text for keyword in HXQ_ROLE_KEYWORDS):
                    signal = bytes_to_tensor(audio_in)
                    audio_emb = spkrec.encode_batch(signal).squeeze(0)
                    logger.info(f"keyword audio_emb")
                    websocket.speaker_verification_sample_emb = audio_emb
                    websocket.speaker_verification_activate = True

        end_time = time.time()
        execution_time = end_time - start_time
        # logger.info(f"async_asr generate speaker_verify execute time：{execution_time}秒")
        if model_punc is not None and len(rec_result["text"]) > 0:
            # logger.info(f"offline, before punc rec_result: {rec_result}")
            rec_result = model_punc.generate(
                input=rec_result["text"], **websocket.status_dict_punc
            )[0]
            # logger.info(f"offline, after punc rec_result: {rec_result}")
        if len(rec_result["text"]) > 0:
            # logger.info(f"offline rec_result: {rec_result}")
            mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            await websocket.send(message)

    else:
        mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
        message = json.dumps(
            {
                "mode": mode,
                "text": "",
                "wav_name": websocket.wav_name,
                "is_final": websocket.is_speaking,
            }
        )
        await websocket.send(message)


async def async_asr_online(websocket, audio_in):
    if len(audio_in) > 0:
        start_time = time.time()

        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get(
            "is_final", False
        ):
            return
            #     websocket.status_dict_asr_online["cache"] = dict()

        rec_result = model_asr_streaming.generate(
            input=audio_in, **websocket.status_dict_asr
        )[0]
        # logger.info(f"async_asr_online: rec_result: {rec_result}")
        text = rec_result["text"]

        # if IS_SPEAKER_VERIFICATION:
        #     if not speaker_verify(websocket, audio_in, text):
        #         return
        end_time = time.time()
        execution_time = end_time - start_time
        # logger.info(f"async_asr_online generate speaker_verify execute time：{execution_time}秒")
        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            await websocket.send(message)


if len(args.certfile) > 0:
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # Generate with Lets Encrypt, copied to this location, chown to current user and 400 permissions
    ssl_cert = args.certfile
    ssl_key = args.keyfile

    ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
    logger.info(f"host: {args.host}, port: {args.port}, ssl_context: {ssl_context}")
    start_server = websockets.serve(
        ws_serve,
        args.host,
        args.port,
        subprotocols=["binary"],
        ping_interval=None,
        ssl=ssl_context,
    )
else:
    logger.info(f"host: {args.host}, port: {args.port}")
    start_server = websockets.serve(
        ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None
    )
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
