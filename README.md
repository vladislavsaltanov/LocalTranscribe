# LocalTranscribe

Локальная расшифровка видео в текст с таймкодами и разделением по спикерам. Всё работает на вашей видеокарте NVIDIA, без облачных сервисов.

- **Русский** (по умолчанию) — [GigaAM-v3](https://github.com/salute-developers/GigaAM)
- **Английский** (`--lang en`) — [Whisper large-v3](https://github.com/openai/whisper) через [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Спикеры** — [pyannote](https://github.com/pyannote/pyannote-audio), количество определяется автоматически

Пример результата:

```
[00:00:00] Speaker 1: Welcome to the show. Today we are talking about medical imaging.

[00:00:07] Speaker 2: Thanks for having me. Medical imaging lets us look inside the human body.
```

## Требования

- Windows, видеокарта NVIDIA с актуальным драйвером
- Python 3.11–3.12, [git](https://git-scm.com/), [ffmpeg](https://ffmpeg.org/) в `PATH`
- Аккаунт на [Hugging Face](https://huggingface.co/) — для моделей pyannote

## Установка

1. Запустите `setup.bat`. Он создаст `.venv`, склонирует GigaAM, поставит зависимости и CUDA-сборку torch. Если в конце `CUDA available: False`, попробуйте другую сборку: `set CUDA_TAG=cu126` и запустите снова.
2. Примите условия использования моделей на Hugging Face (кнопка на странице модели):
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Создайте токен в [настройках Hugging Face](https://huggingface.co/settings/tokens) (достаточно права Read) и сохраните его:

   ```
   setx HF_TOKEN hf_xxx
   ```

   После этого откройте терминал заново.

При первом запуске скачиваются модели: GigaAM для русского, около 3 ГБ Whisper для английского.

## Запуск

```
run.bat video.mp4 -o video.txt                          # русский
run.bat video.mp4 -o video.txt --lang en                # английский
run.bat video.mp4 -o video.txt --lang en --speakers 2   # точное число спикеров
run.bat video.mp4 -o video.txt --speakers 1             # без разделения по спикерам
```

| Флаг | Что делает |
|---|---|
| `-o FILE` | Файл результата (по умолчанию `transcript.txt`) |
| `--lang ru\|en` | Язык видео (по умолчанию `ru`) |
| `--speakers N` | Число спикеров. По умолчанию определяется само; `1` — без разделения |
| `--model NAME` | Модель распознавания (по умолчанию `v3_e2e_rnnt` для ru, `large-v3` для en; для скорости — `large-v3-turbo`) |
| `--keep-fillers` | Не убирать «э-э», «м-м», um, uh и т.п. |
| `--gigaam-vad` | Штатная нарезка GigaAM вместо нарезки по паузам (только ru) |

## Результат

- Реплики с таймкодом и меткой спикера (`Спикер 1` / `Speaker 1`).
- Если спикер один, разделить не удалось или для английского не задан `HF_TOKEN`, текст пишется без меток, абзацами по паузам. Распознанный текст сохраняется в любом случае.
- Рядом сохраняется `<файл>.turns.txt` — сырая разметка спикеров, чтобы при плохом результате понять, ошиблась ли диаризация.

Если автоопределение ошиблось с числом спикеров, укажите его через `--speakers N`.

## Лицензия

Код — [MIT](LICENSE). Модели распространяются на своих условиях: GigaAM и Whisper — MIT, модели pyannote требуют принять условия на Hugging Face.
