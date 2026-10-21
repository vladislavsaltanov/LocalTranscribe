#!/usr/bin/env python3
"""Транскрипция интервью: GigaAM-v3 (русский) или Whisper (английский) по словам + pyannote (спикеры).

Запуск:
    set HF_TOKEN=hf_xxx
    run.bat interview.mp4 -o interview.txt
    run.bat interview.mp4 -o interview.txt --lang en --speakers 2
"""
import argparse
import contextlib
import io
import os
import re
import subprocess
import tempfile
import warnings

# Безвредные предупреждения pyannote/torch - прячем, чтобы не мусорили в консоли
warnings.filterwarnings("ignore", message="(?s).*TensorFloat-32")
warnings.filterwarnings("ignore", message="(?s).*degrees of freedom is <= 0")

import numpy as np
import soundfile as sf
import torch  # заодно грузит cuBLAS/cuDNN - без них faster-whisper не запустится на GPU
import gigaam
from gigaam import vad_utils
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from tqdm import tqdm

CHUNK_S = 300  # длина куска для распознавания, сек (от неё зависит частота обновления прогресса)
BAR_FMT = "{l_bar}{bar}| {n:.0f}/{total:.0f} с аудио [{elapsed}<{remaining}]"


def hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def extract_audio(video: str, wav: str) -> None:
    # 16 кГц, моно - формат, который ждут обе модели
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def find_cuts(data: np.ndarray, sr: int, chunk_s: int = CHUNK_S, win_s: int = 10) -> list:
    """Границы кусков (в сэмплах). Каждая граница сдвигается в самое тихое место
    в пределах +-win_s секунд, чтобы не резать слова."""
    frame = int(0.1 * sr)
    cuts = [0]
    pos = chunk_s * sr
    while pos + chunk_s * sr // 2 < len(data):  # не оставляем крошечный хвост
        lo, hi = pos - win_s * sr, pos + win_s * sr
        seg = data[lo:hi]
        n = len(seg) // frame
        energy = (seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
        cut = lo + int(energy.argmin()) * frame + frame // 2
        cuts.append(cut)
        pos = cut + chunk_s * sr
    cuts.append(len(data))
    return cuts


def plan_chunks(regions, max_d=24.0, min_d=6.0, pad=0.2, total=None) -> list:
    """Из областей речи (start, end) делает куски для модели не длиннее ~max_d секунд.
    Режет по самым длинным паузам, а не по первой паузе после 15 с, как GigaAM по умолчанию."""
    flat = []  # непрерывную речь длиннее max_d делим поровну
    for s, e in regions:
        n = int((e - s) / max_d) + 1 if e - s > max_d else 1
        step = (e - s) / n
        flat += [(s + k * step, s + (k + 1) * step) for k in range(n)]

    chunks, i = [], 0
    while i < len(flat):
        start, best, k = flat[i][0], None, i
        while k < len(flat) and flat[k][1] - start <= max_d:
            if k + 1 < len(flat) and flat[k][1] - start >= min_d:
                gap = flat[k + 1][0] - flat[k][1]
                if best is None or gap >= best[0]:  # при равных паузах берём более длинный кусок
                    best = (gap, k)
            k += 1
        if k == len(flat):
            cut = k - 1  # всё оставшееся влезает в один кусок
        elif best is not None:
            cut = best[1]
        else:
            cut = max(k - 1, i)
        chunks.append((start, flat[cut][1]))
        i = cut + 1

    if total is None and chunks:
        total = chunks[-1][1] + pad
    spans = []
    for n, (a, b) in enumerate(chunks):
        prev_end = chunks[n - 1][1] if n else 0.0
        next_start = chunks[n + 1][0] if n + 1 < len(chunks) else total
        # небольшой запас по краям, чтобы VAD не обрезал начала и концы слов; не заходим дальше середины паузы
        a2 = max(0.0, a - min(pad, max(0.0, (a - prev_end) / 2)))
        b2 = min(total, b + min(pad, max(0.0, (next_start - b) / 2)))
        if b2 - a2 >= 0.2:
            spans.append((a2, b2))
    return spans


def pause_aware_segments(wav_file, sr, device=torch.device("cpu"), **_):
    """Подмена нарезки GigaAM (vad_utils.segment_audio_file): возвращает (куски аудио, границы)."""
    audio = gigaam.load_audio(wav_file)
    total = audio.shape[0] / sr
    vad = vad_utils.get_pipeline(device)
    regions = [(max(0.0, r.start), min(total, r.end)) for r in vad(wav_file).get_timeline().support()]
    spans = plan_chunks(regions, total=total)
    return [audio[int(a * sr):int(b * sr)] for a, b in spans], spans


# Вокализованные паузы: э-э, эээ, эм, м-м-м, хмм, а-а, и-и-и, ы-ы. Обычные слова (а, и, ну, вот, как бы) не трогаем.
# В английском: uh, um, erm, hmm, mm; uh-huh и mhm значат «да» - их оставляем.
FILLER_RE = {
    "ru": re.compile(
        r"^(?:э+(?:-э+)*|э-?м+(?:-м+)*|м{2,}(?:-м+)*|м(?:-м+)+|х+м+(?:-м+)*"
        r"|а(?:-а)+|а{2,}|и(?:-и)+|и{2,}|ы+(?:-ы+)*)$"
    ),
    "en": re.compile(r"^(?:u+[hm]+|e+r+m+|h+m+|m{2,})$"),
}
PUNCT = ".,!?…:;«»\"'()—–-"
SENT_END = (".", "!", "?", "…")


def cap_first(word: str) -> str:
    for i, ch in enumerate(word):
        if ch.isalpha():
            return word[:i] + ch.upper() + word[i + 1:]
    return word


def drop_fillers(words: list, filler_re) -> list:
    """Убирает вокализованные паузы, сохраняя точку/вопрос, которые они несли, и заглавную в начале предложения."""
    out, need_cap = [], False
    for w in words:
        core = w.strip(PUNCT).lower()
        if core and filler_re.match(core):
            m = re.search(r"[.!?]+$", w)
            if out and m and not out[-1].endswith(SENT_END):
                out[-1] = out[-1].rstrip(",;:") + m.group(0)
            if not out or out[-1].endswith(SENT_END):
                need_cap = True  # филлер стоял в начале предложения - следующее слово станет первым
            continue
        out.append(cap_first(w) if need_cap else w)
        need_cap = False
    return out


def transcribe(wav: str, model_name: str, tmp: str, pause_vad: bool = True) -> list:
    """Возвращает слова [(start, end, text), ...] в секундах от начала записи."""
    data, sr = sf.read(wav, dtype="float32")
    cuts = find_cuts(data, sr)

    print("Загрузка модели распознавания...")
    asr = gigaam.load_model(model_name)
    if pause_vad:  # transcribe_longform берёт segment_audio_file из vad_utils в момент вызова
        vad_utils.segment_audio_file = pause_aware_segments

    words = []
    bar = tqdm(total=len(data) / sr, unit="s", desc="Распознавание", bar_format=BAR_FMT)
    for a, b in zip(cuts[:-1], cuts[1:]):
        part = os.path.join(tmp, "part.wav")
        sf.write(part, data[a:b], sr)
        offset = a / sr
        last = None
        with contextlib.redirect_stdout(io.StringIO()):  # GigaAM печатает служебную статистику
            result = asr.transcribe_longform(part, word_timestamps=True)
        for seg in result:
            if seg.words:
                for w in seg.words:
                    words.append((w.start + offset, w.end + offset, w.text))
            elif seg.text.strip():  # запасной вариант: сегмент без слов целиком
                words.append((seg.start + offset, seg.end + offset, seg.text.strip()))
            if seg.text.strip():
                last = seg
        bar.update((b - a) / sr)
        if last is not None:  # последняя распознанная фраза - чтобы видеть, что оно живое
            tqdm.write(f"  [{hms(last.start + offset)}] {last.text.strip()[:70]}")
    bar.close()
    return words


def transcribe_en(wav: str, model_name: str) -> list:
    """Английский через Whisper (faster-whisper). Возвращает слова [(start, end, text), ...] в секундах."""
    data, _ = sf.read(wav, dtype="float32")
    print("Загрузка модели распознавания...")
    cuda = torch.cuda.is_available()
    asr = WhisperModel(model_name, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
    # vad_filter пропускает тишину, на которой Whisper галлюцинирует; без опоры на прошлый текст он не зацикливается
    segments, info = asr.transcribe(data, language="en", word_timestamps=True, vad_filter=True,
                                    condition_on_previous_text=False)
    words, next_report = [], CHUNK_S
    bar = tqdm(total=info.duration, unit="s", desc="Распознавание", bar_format=BAR_FMT)
    for seg in segments:  # генератор: распознавание идёт по мере чтения
        for w in seg.words:
            if words and not w.word.startswith(" "):  # кусок слова без пробела (X + -rays) - клеим к предыдущему
                words[-1] = (words[-1][0], w.end, words[-1][2] + w.word.strip())
            elif w.word.strip():
                words.append((w.start, w.end, w.word.strip()))
        bar.update(seg.end - bar.n)
        if seg.end >= next_report:  # свежая фраза раз в CHUNK_S секунд аудио - чтобы видеть, что оно живое
            tqdm.write(f"  [{hms(seg.start)}] {seg.text.strip()[:70]}")
            next_report = seg.end + CHUNK_S
    bar.update(info.duration - bar.n)
    bar.close()
    return words


def load_pipeline(name: str, token: str):
    try:
        try:
            pipe = Pipeline.from_pretrained(name, token=token)
        except TypeError:  # старые версии pyannote принимают use_auth_token
            pipe = Pipeline.from_pretrained(name, use_auth_token=token)
    except Exception as e:
        pipe, err = None, e
    else:
        err = None
    if pipe is None:
        raise RuntimeError(
            f"Не удалось загрузить {name}: {err or 'нет доступа'}\n"
            "Проверьте HF_TOKEN и примите условия на странице модели на huggingface.co."
        )
    return pipe


def diarize(wav: str, token: str, num_speakers: int, model: str):
    pipe = load_pipeline(model, token)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))

    # Грузим аудио сами, чтобы не зависеть от декодера внутри pyannote
    data, sr = sf.read(wav, dtype="float32")
    audio = {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": sr}
    with ProgressHook() as hook:  # показывает прогресс по этапам диаризации
        out = pipe(audio, num_speakers=num_speakers, hook=hook)
    # pyannote 4.x: объект с полями; exclusive - без перекрытий, удобнее для сопоставления со словами
    if getattr(out, "exclusive_speaker_diarization", None) is not None:
        ann = out.exclusive_speaker_diarization
    elif getattr(out, "speaker_diarization", None) is not None:
        ann = out.speaker_diarization
    else:  # 3.x: сразу Annotation
        ann = out
    return [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]


def speaker_of(start: float, end: float, turns) -> str:
    """Спикер с максимальным пересечением со словом; если пересечений нет - ближайший по времени."""
    overlap = {}
    for s, e, spk in turns:
        o = min(end, e) - max(start, s)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0.0) + o
    if overlap:
        return max(overlap, key=overlap.get)
    mid = (start + end) / 2
    return min(turns, key=lambda t: max(t[0] - mid, mid - t[1], 0.0))[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--output", default="transcript.txt")
    ap.add_argument("--lang", choices=["ru", "en"], default="ru", help="язык видео: ru - GigaAM, en - Whisper")
    ap.add_argument("--speakers", type=int, help="число спикеров; по умолчанию определяется само, 1 - без разделения")
    ap.add_argument("--model", help="модель распознавания; по умолчанию v3_e2e_rnnt для ru, large-v3 для en")
    ap.add_argument("--diar-model", default="pyannote/speaker-diarization-community-1")
    ap.add_argument("--keep-fillers", action="store_true", help="не убирать э-э, м-м-м, um, uh и т.п.")
    ap.add_argument("--gigaam-vad", action="store_true", help="штатная нарезка GigaAM вместо нарезки по паузам (ru)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        if args.lang == "ru":  # GigaAM читает токен для нарезки длинного аудио из этой же переменной окружения
            raise SystemExit("Задайте HF_TOKEN (нужен для pyannote).")
        print("HF_TOKEN не задан - текст будет без спикеров.")

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        print("[1/4] Извлечение звука...")
        extract_audio(args.video, wav)

        print("[2/4] Распознавание речи")
        if args.lang == "en":
            words = transcribe_en(wav, args.model or "large-v3")
        else:
            words = transcribe(wav, args.model or "v3_e2e_rnnt", tmp, pause_vad=not args.gigaam_vad)

        turns = []
        if token and args.speakers != 1:
            print("[3/4] Разделение по спикерам")
            try:
                turns = diarize(wav, token, args.speakers, args.diar_model)
            except Exception as e:  # нет доступа к модели, не хватило памяти и т.п. - текст всё равно сохраняем
                print(f"  Ошибка: {e}")
            if len({spk for _, _, spk in turns}) < 2:  # с одним спикером метки не нужны
                print("  Найден один спикер или ни одного - текст будет без меток.")
                turns = []

    print("[4/4] Сборка файла...")
    # Сырая разметка спикеров - чтобы при плохом результате понять, виновата диаризация или сопоставление
    if turns:
        with open(args.output + ".turns.txt", "w", encoding="utf-8") as f:
            for s, e, spk in sorted(turns):
                f.write(f"{hms(s)} - {hms(e)}  {spk}\n")

    # Имена по порядку первого появления: Спикер 1, Спикер 2
    label = "Speaker" if args.lang == "en" else "Спикер"
    names = {}
    runs = []  # [start, speaker, [слова]]
    prev_end = 0.0
    for start, end, text in words:
        if turns:  # слова подряд от одного спикера - одна реплика
            name = names.setdefault(speaker_of(start, end, turns), f"{label} {len(names) + 1}")
            new = not runs or runs[-1][1] != name
        else:  # без спикеров - новый абзац на конце предложения после паузы от 2 с или если абзац длиннее минуты
            name = None
            new = not runs or (runs[-1][2][-1].endswith(SENT_END)
                               and (start - prev_end >= 2 or start - runs[-1][0] >= 60))
        if new:
            runs.append([start, name, [text]])
        else:
            runs[-1][2].append(text)
        prev_end = end

    lines = []
    for start, name, ws in runs:
        if not args.keep_fillers:
            ws = drop_fillers(ws, FILLER_RE[args.lang])
        if not ws:  # реплика состояла только из «э-э»
            continue
        if name and lines and lines[-1][1] == name:  # после удаления соседние реплики одного спикера склеиваются
            lines[-1][2].extend(ws)
        else:
            lines.append([start, name, ws])

    with open(args.output, "w", encoding="utf-8") as f:
        for start, name, ws in lines:
            prefix = f"{name}: " if name else ""
            f.write(f"[{hms(start)}] {prefix}{' '.join(ws)}\n\n")
    print(f"Готово: {args.output}")


if __name__ == "__main__":
    main()
