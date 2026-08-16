# Personalized real-time privacy filter

Real-time фильтр для камеры: зарегистрированный владелец остаётся видимым, остальные обнаруженные лица скрываются пикселизацией.

Проект использует YOLO11n-face, IResNet100@Glint360K из набора InsightFace
AntelopeV2, ONNX Runtime и OpenCV. Лендмарки и дополнительные landmark-модели
не используются. На Apple Silicon автоматически используется CoreML. На
Windows автоматически используется DirectML с совместимой NVIDIA, AMD или
Intel GPU, а при недоступности GPU — CPU.

## Клонирование

```bash
git clone https://github.com/korosaii/Personalized-real-time-privacy-filter.git
cd Personalized-real-time-privacy-filter
```

## Установка

Создайте окружение на любой операционной системе:

```bash
python -m venv .venv
```

macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Установите проект:

```bash
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt
```

На Windows пакет DirectML устанавливается автоматически. Выбирать модель видеокарты или provider вручную не нужно.

Разрешите доступ к камере для Terminal или VS Code в настройках приватности операционной системы.

## Модели

```text
models/
├── detector/
│   ├── yolov11n-face.onnx
│   ├── yolov11n-face-pose.onnx
│   └── yolov11n-face-pose-roll90.onnx
└── recognition/
    ├── iresnet_r34_glint360k.onnx
    ├── iresnet_r100_glint360k.onnx
    └── webface_r50.onnx
```

Runtime использует статический YOLO11 ONNX с входом `640×640` и выбранную recognition-модель с входом `112×112`. Детектор `yolo11` и recognition-модель `r34-glint360k` используются по умолчанию. Детекторы `yolo11-pose` и `yolo11-pose-roll90` выдают bbox и пять точек лица за один inference. Вариант `yolo11-pose-roll90` дообучен на WIDER FACE и WFLW с поворотами до ±90°.

Доступные имена моделей: `r34-glint360k`, `r100-glint360k`, `r50-webface600k`. Официальной R100@WebFace600K в InsightFace Model Zoo нет; для R100 используется Glint360K.


## Регистрация владельца

Добавьте одну или несколько фотографий в:

```text
data/photos/owner/
```

Команда на macOS и Windows:

```bash
privacy-enroll owner data/photos/owner
```

Для регистрации с выравниванием по глазам, носу и углам рта создайте отдельный template:

```bash
privacy-enroll owner data/photos/owner --detector yolo11-pose --output data/enrollments/owner-pose.npz
```

Для честного сравнения экспериментального detector, дообученного на больших наклонах, создайте отдельный template:

```bash
privacy-enroll owner data/photos/owner --detector yolo11-pose-roll90 --output data/enrollments/owner-pose-roll90.npz
```

Для сравнения моделей создавайте отдельный template для каждой:

```bash
privacy-enroll owner data/photos/owner --model r34-glint360k --output data/enrollments/owner-r34.npz
privacy-enroll owner data/photos/owner --model r100-glint360k --output data/enrollments/owner-r100.npz
privacy-enroll owner data/photos/owner --model r50-webface600k --output data/enrollments/owner-r50-w600k.npz
```

По умолчанию каждая фотография даёт один обычный embedding. Для теста с поворотами создайте отдельный template:

```bash
privacy-enroll owner data/photos/owner --model r100-glint360k --rotations --output data/enrollments/owner-r100-rot.npz
privacy-enroll owner data/photos/owner --model r50-webface600k --rotations --output data/enrollments/owner-r50-w600k-rot.npz
```

Biometric template сохраняется в `data/enrollments/owner.npz`. Исходные фотографии и enrollment data исключены из Git.

Без `--rotations` embeddings обычных фотографий усредняются в один L2-normalized centroid. С `--rotations` для каждого фото создаются шесть embedding для `0°`, `30°`, `90°`, `180°`, `270°` и `330°`, затем строится шесть centroid. Runtime берёт максимальное сходство по активным centroid.

Порог авторизации по умолчанию: `0.35`.

## Запуск

```bash
privacy-recognize
```

Запуск Pose-детектора требует template, созданный с тем же detector mode:

```bash
privacy-recognize --detector yolo11-pose --template data/enrollments/owner-pose.npz
```

Запуск экспериментального варианта с большими наклонами:

```bash
privacy-recognize --detector yolo11-pose-roll90 --template data/enrollments/owner-pose-roll90.npz
```

В этом режиме перед recognition лицо геометрически приводится к ArcFace `112×112` по пяти точкам. Выравнивание исправляет положение, масштаб и наклон головы в плоскости изображения. Оно не выполняет 3D-фронтализацию сильного профиля.

По умолчанию используется официальный Ultralytics ByteTrack. Доступны три режима:

```bash
privacy-recognize --tracker bytetrack
privacy-recognize --tracker botsort
privacy-recognize --tracker iou
```

`bytetrack` и `botsort` используют реализацию из пакета Ultralytics. `iou` оставлен как лёгкий локальный baseline. BoT-SORT работает без дополнительной ReID-модели. При потере или неуверенном продолжении трека состояние `AUTHORIZED` сбрасывается, и лицо снова скрывается до recognition.

Для ByteTrack и BoT-SORT detector threshold автоматически равен `0.10`, чтобы официальный трекер получал как уверенные, так и слабые detections. Для локального IoU baseline автоматически используется `0.25`. Явное значение можно передать через `--detector-threshold`.

Запуск конкретной модели с её template:

```bash
privacy-recognize --model r34-glint360k --template data/enrollments/owner-r34.npz
privacy-recognize --model r100-glint360k --template data/enrollments/owner-r100.npz
privacy-recognize --model r50-webface600k --template data/enrollments/owner-r50-w600k.npz
```

Для template, созданного с поворотами, добавьте тот же флаг:

```bash
privacy-recognize --model r100-glint360k --rotations --template data/enrollments/owner-r100-rot.npz
privacy-recognize --model r50-webface600k --rotations --template data/enrollments/owner-r50-w600k-rot.npz
```

Изображение камеры зеркалится по умолчанию. Для обычного незеркального отображения:

```bash
privacy-recognize --no-mirror
```

Лица с bbox меньше `80 px` остаются скрытыми и не отправляются в recognition. Перед первой проверкой лицо должно полностью находиться внутри кадра, а трек — быть стабильным три кадра. Новый подходящий трек проходит три проверки. После авторизации стабильный трек сохраняет состояние без постоянного recognition, в том числе при частичном выходе за край кадра. UNKNOWN повторно проверяется через `30`, `60`, `120`, `240` и затем каждые `300` кадров. Новый или неуверенный трек проверяется сразу. Авторизация сохраняется при уменьшении лица до `56 px`.

Неавторизованное лицо пикселизируется внутри овала, вписанного непосредственно
в bbox. Bbox не расширяется, а его углы остаются без пикселизации.

Освещение оценивается по padding-кольцу размером 25% вокруг bbox без пикселей
самого лица и по сохранности информации внутри face crop. Метрики сглаживаются
по треку. Используются режимы `NORMAL`, `LOW_LIGHT` и `OVEREXPOSED` (засвет).
По умолчанию предполагается, что при enrollment не было тёмных или засвеченных
фотографий: в двух ухудшенных режимах порог повышается на `0.10`, а пограничные
совпадения остаются `UNKNOWN`. Если такие фотографии были добавлены при
enrollment, запускайте распознавание с флагом:

```bash
privacy-recognize --enrollment-has-difficult-lighting
```

Тогда в `LOW_LIGHT` и `OVEREXPOSED` используется тот же порог, что и в
`NORMAL`. Величину повышения без флага можно изменить через
`--difficult-lighting-threshold-increase`.
При ухудшении освещения авторизованный трек сразу скрывается и проходит
повторную проверку. Количество лиц в каждом режиме показывается в верхнем
оверлее; показатели освещения рядом с bbox не выводятся.

Завершение: `Q`, `Esc` или `Ctrl+C`. Последний benchmark сохраняется локально в `benchmarks/latest.json`.

Дополнительные параметры:

```bash
privacy-recognize --help
privacy-enroll --help
inspect-models --help
```

## Offline-обработка видео через Grounding DINO + SAM 2.1

По умолчанию offline-режим использует публичные веса
`IDEA-Research/grounding-dino-tiny` и `facebook/sam2.1-hiera-small`. Grounding
DINO находит объекты по тексту, а SAM 2.1 сегментирует их и отслеживает между
повторными детекциями:

```powershell
privacy-recognize --offline-video `
  --video-path data/videos/input.mp4 `
  --video-prompt "face" `
  --video-prompt "license plate" `
  --video-output data/videos/input.redacted.mp4
```

Если `--video-output` не указан, результат сохраняется рядом с исходником как
`<имя>.redacted.mp4`. Веса обеих моделей загружаются из Hugging Face
автоматически и не требуют заявки или токена. Необходимые дополнительные пакеты:

```powershell
python -m pip install -e ".[grounded-video]"
git clone https://github.com/facebookresearch/sam2.git C:\yandex\sam2
python -m pip install -e C:\yandex\sam2
```

PyTorch устанавливайте отдельно со сборкой под имеющуюся CUDA. Без CUDA режим
тоже запускается, но будет очень медленным. По умолчанию DINO повторно ищет
объекты каждые 25 кадров; интервал меняется через
`--grounding-redetect-interval`. Пороги задаются параметрами
`--grounding-box-threshold` и `--grounding-text-threshold`.
Интервал одновременно является размером потокового SAM 2.1 чанка: в памяти
находятся только эти кадры, а не всё видео. Устройство выбирается автоматически;
его можно зафиксировать через `--offline-device cpu` или
`--offline-device cuda`.
Для инференса 4K-кадры по умолчанию уменьшаются до 1280 px по длинной стороне,
после чего маска возвращается в исходное разрешение. Размер меняется через
`--video-inference-max-side`; значение `0` отключает уменьшение.
Для короткого пробного запуска можно ограничить обработку, например
`--max-frames 30`. Чтобы изменить разрешение готового файла, укажите, например,
`--video-output-size 1920x1080`. Маски при этом масштабируются вместе с кадрами,
а исходное видео не изменяется.

Параметр `--video-prompt` можно повторить несколько раз. Все классы передаются
Grounding DINO одновременно, а маски объединяются перед пикселизацией. Выходной
MP4 пока создаётся без аудиодорожки. Если SAM 2.1 не вернул маску для ожидаемого
кадра, такой кадр полностью пикселизируется.

Предыдущий SAM 3 backend сохранён и включается явно через
`--offline-backend sam3`.
