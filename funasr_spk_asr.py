import datetime

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from moviepy import VideoFileClip


def extract_audio(input_video_path, output_audio_path):
    video = VideoFileClip(input_video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)

if __name__ == "__main__":
    # audio_in = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_speaker_demo.wav"
    video_mp4_test = "https://www.test.com/1650544532622.mp4"

    audio_in = f"output_audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"

    extract_audio(video_mp4_test, audio_in)

    output_dir = "./results"
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
        model_revision="v2.0.4",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_model_revision="v2.0.4",
        punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        punc_model_revision="v2.0.4",
        output_dir=output_dir,
    )
    rec_result = inference_pipeline(
        audio_in, batch_size_s=300, batch_size_token_threshold_s=40
    )
    print(rec_result)

    sentence_info = rec_result[0]["sentence_info"]

    results = []

    for sentence in sentence_info:
        speaker = f"speaker-{sentence['spk']}"
        text = sentence['text']
        results.append({
            "speaker": speaker,
            "text": text
        })

    # 保存为txt文件
    output_txt = f"doctor-patient-output_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['speaker']}: {r['text']}\n")

