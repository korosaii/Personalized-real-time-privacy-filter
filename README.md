# Personalized real-time privacy filter

Фильтр приватности для камеры и видео. В основном режиме зарегистрированные
владельцы остаются видимыми, остальные лица скрываются. Также поддерживается
скрытие объектов по текстовому описанию или reference-изображениям.

Основной face pipeline использует:

- дообученный `YOLO11Face` с bbox и пятью лицевыми лендмарками;
- выравнивание лица по пяти точкам перед извлечением embedding;
- `IResNet50` для распознавания;
- лёгкий локальный IoU tracker для сопровождения лиц;
- ONNX Runtime и OpenCV.

На Windows `auto` выбирает DirectML при наличии совместимого GPU, на Apple
Silicon — CoreML, иначе используется CPU.

## Установка

```bash
git clone https://github.com/korosaii/Personalized-real-time-privacy-filter.git
cd Personalized-real-time-privacy-filter
python -m venv .venv
```

Активация окружения:

```powershell
# Windows
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Полная установка:

```bash
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt
```

Минимальная установка только для face privacy:

```bash
python -m pip install -e .
```

Дополнительные наборы зависимостей:

| Возможность | Команда |
|---|---|
| Виртуальная камера | `python -m pip install -e ".[virtual-camera]"` |
| Offline text-prompt | `python -m pip install -e ".[grounded-video]"` |
| Realtime image-prompt | `python -m pip install -e ".[image-prompt]"` |
| Экспорт OpenVINO INT8 | `python -m pip install -e ".[image-prompt,quantization]"` |

Разрешите Terminal или IDE доступ к камере в настройках операционной системы.

## Модели

Для основного режима нужны файлы:

```text
models/
├── detector/
│   ├── yolov11n-face-pose-roll90.onnx
│   └── yolov5-face.onnx
├── recognition/
│   └── webface_r50.onnx
└── yoloe/
    └── yoloe-26n-seg.pt
```

Несмотря на имя файла detector-весов, это основная дообученная модель проекта,
а не отдельный экспериментальный режим. В CLI она называется `yolo11face`.
Дефолтная recognition-модель называется `iresnet50` и загружается из
`models/recognition/webface_r50.onnx`.

Веса дефолтного detector можно скачать из
[папки проекта на Google Drive](https://drive.google.com/drive/folders/1aKJjOsLMXJXPThVyqeMCkzfftczEvWYX)
и положить в `models/detector/yolov11n-face-pose-roll90.onnx`.
Альтернативный `yolov5-face` также выдаёт bbox и пять
лендмарок и доступен через `--detector yolov5-face`.

Свои ONNX-модели можно передать путём через `--detector` и `--model`. Detector
для основного сценария должен возвращать bbox и пять лендмарков, recognition
model должна принимать `1×3×112×112` и возвращать 512-мерный embedding.

Проверить метаданные файлов:

```bash
inspect-models models/detector/yolov11n-face-pose-roll90.onnx models/recognition/webface_r50.onnx
```

## Регистрация владельцев

Регистрация по умолчанию автоматическая. Создайте отдельную подпапку для каждого
владельца:

```text
data/photos/
├── alice/
│   ├── 01.jpg
│   └── 02.jpg
└── bob/
    ├── 01.jpg
    └── 02.jpg
```

После запуска `privacy-recognize` фотографии проверяются, лица выравниваются по
глазам, носу и уголкам рта, затем templates сохраняются в `data/enrollments/`.
То есть да: при дефолтной регистрации выравнивание включено автоматически.
Фотография с недостаточно уверенными landmarks не должна попадать в template.
Исходные фотографии в template не копируются.

Желательно использовать несколько чётких фотографий одного человека с разными
выражениями, освещением и небольшими изменениями ракурса. На каждой фотографии
должно быть ровно одно лицо размером не менее 80 px.

Ручная регистрация — опциональный способ заранее создать template:

```bash
privacy-enroll alice data/photos/alice
privacy-enroll bob data/photos/bob
```

Явное подключение готовых templates:

```powershell
privacy-recognize `
  --template data/enrollments/alice.npz `
  --template data/enrollments/bob.npz
```

Можно передать каталог с `.npz`:

```bash
privacy-recognize --template data/enrollments/family
```

Наличие `--template` отключает автоматическую регистрацию для текущего запуска.
`--no-auto-enroll` также позволяет использовать уже готовый
`data/enrollments/owner.npz`. Опция регистрации с искусственными поворотами
удалена: наклон в плоскости кадра компенсируют detector landmarks и ArcFace
alignment.

Основные флаги ручной регистрации:

| Флаг | По умолчанию | Назначение |
|---|---:|---|
| `--output` | `data/enrollments/<name>.npz` | Путь результата. |
| `--detector` | `yolo11face` | Detector alias или путь к ONNX. |
| `--model` | `iresnet50` | Recognition alias или путь к ONNX. |
| `--provider` | `auto` | ONNX Runtime provider. |
| `--threshold` | `0.35` | Порог, сохраняемый в template. |
| `--min-face-size` | `80` | Минимальная сторона лица. |
| `--min-sharpness` | `25` | Минимальная резкость face crop. |

## Face privacy

Запуск с веб-камерой:

```bash
privacy-recognize
```

Без дополнительных флагов используются `yolo11face`, `iresnet50`,
автоматическая регистрация, локальный IoU tracker и зеркальное отображение камеры.
Неавторизованные лица скрываются fail-closed: при ошибке или потере уверенного
track лицо не остаётся открытым.

Обработка видео тем же realtime pipeline:

```powershell
privacy-recognize --realtime-video `
  --video-path data/videos/input.mp4 `
  --video-output data/videos/output.mp4
```

По умолчанию используется локальный IoU tracker. ByteTrack и BoT-SORT можно
включить явно:

```bash
privacy-recognize --tracker bytetrack
privacy-recognize --tracker botsort
privacy-recognize --tracker iou
```

Для ByteTrack и BoT-SORT detector threshold по умолчанию равен `0.10`, для
локального IoU tracker — `0.25`.

### Оверлеи и Zoom

Отключить bbox и статистику, оставив только скрытие лиц:

```bash
privacy-recognize --no-bboxes --no-statistics
```

Флаги независимы: можно оставить только bbox или только текстовую диагностику.

Камера зеркалится по умолчанию. При `--no-mirror` кадр передаётся без отражения,
но текст статистики заранее отражается отдельно. Zoom зеркалит локальное
изображение ещё раз, поэтому надписи в его preview читаются корректно. Bbox и
маски при этом остаются на своих объектах. Если текст не нужен, используйте
`--no-statistics`.

```bash
privacy-recognize --no-mirror
```

### Виртуальная камера

```powershell
privacy-recognize --virtual-camera --no-preview
```

На Windows установите OBS Studio, чтобы появилось устройство `OBS Virtual
Camera`, затем выберите его в Zoom, браузере или мессенджере. Исходный кадр в
виртуальную камеру не отправляется — только обработанный результат.

Основные face-флаги:

| Флаг | По умолчанию | Назначение |
|---|---:|---|
| `--detector` | `yolo11face` | Landmark detector. |
| `--model` | `iresnet50` | Recognition model. |
| `--tracker` | `iou` | `iou`, опциональные `bytetrack` или `botsort`. |
| `--minimum-recognition-face-size` | `80` | Минимальное лицо для recognition. |
| `--minimum-authorized-face-size` | `56` | Минимальный размер уже авторизованного лица. |
| `--confirmations` | `3` | Совпадения подряд до открытия лица. |
| `--mirror` / `--no-mirror` | камера: on, файл: off | Отражение входного кадра. |
| `--bboxes` / `--no-bboxes` | on | Bbox поверх результата. |
| `--statistics` / `--no-statistics` | on | FPS, latency и подписи tracks. |
| `--preview` / `--no-preview` | on | Локальное окно. |
| `--virtual-camera` | off | Публикация обработанного потока. |
| `--benchmark-out` | `benchmarks/latest.json` | Runtime-метрики в JSON. |

Полный список:

```bash
privacy-recognize --help
privacy-enroll --help
```

## Realtime image-prompt: SAM + YOLOE

Этот режим скрывает объекты, похожие на reference-изображения. SAM 2 один раз
выделяет foreground на каждом reference. YOLOE запускается на каждом кадре, а
временные ID связываются лёгким bbox IoU association. EdgeTAM не используется.

По умолчанию исходным checkpoint служит `models/yoloe/yoloe-26n-seg.pt`.
Pipeline вычисляет SHA-256 от фактических cropped reference, бинарных
SAM-масок, группировки классов и размеров YOLOE. Если в `.cache/yoloe/int8`
уже есть модель с таким fingerprint, она используется сразу. Если reference
изменился, автоматически создаётся новая квантизованная OpenVINO INT8-модель.
INT8-артефакты являются локальным кешем и в Git не добавляются.

Для первого запуска с новым объектом нужны исходный checkpoint
`models/yoloe/yoloe-26n-seg.pt` и calibration YAML. Стандартную DAVIS
calibration-выборку можно подготовить так:

```powershell
python benchmarks/prepare_davis_int8_calibration.py `
  --output outputs/int8_calibration_davis_train
```

Первый запуск нового reference занимает около 2–3 минут на проверенном CPU;
следующие запуски используют content-addressed INT8 cache без повторной
квантизации:

```powershell
privacy-recognize --image-prompt-video `
  --reference-image data/references/object.jpg
```

Для немедленного FP32-запуска без INT8 export явно передайте исходную `.pt`
модель: `--image-yolo-model models/yoloe/yoloe-26n-seg.pt`.

```powershell
privacy-recognize --image-prompt-video `
  --reference-image data/references/object.jpg `
  --image-redaction blur `
  --image-sam-mask-output-dir outputs/reference-sam-masks
```

При `--image-sam-mask-output-dir` foreground-маска, которую SAM 2 строит для
каждого reference-изображения, сохраняется как чёрно-белый PNG
(`000_object.png`, ...): белым отмечен выбранный объект, чёрным — фон. Это та
самая маска, которая затем передаётся в YOLOE как visual prompt.

Каталог считается одним классом с несколькими ракурсами:

```powershell
privacy-recognize --image-prompt-video `
  --reference-image data/references/backpack `
  --virtual-camera
```

Для входного файла нужен выходной путь:

```powershell
privacy-recognize --image-prompt-video `
  --reference-image data/references/object.jpg `
  --video-path data/videos/input.mp4 `
  --video-output data/videos/object-redacted.mp4
```

`--no-image-tracker` сохранён как совместимый флаг. Он означает IoU association;
после удаления EdgeTAM это и так единственный режим, поэтому флаг не меняет
результат.

| Флаг | По умолчанию | Назначение |
|---|---:|---|
| `--image-yolo-model` | `models/yoloe/yoloe-26n-seg.pt` | Исходные YOLOE-веса; при включённой автоквантизации запускается INT8 cache flow. |
| `--image-yolo-auto-quantize` | on | Сравнить SHA-256 reference и автоматически создать INT8 при несовпадении. |
| `--image-yolo-source-model` | `models/yoloe/yoloe-26n-seg.pt` | Исходные FP32-веса для нового export. |
| `--image-int8-calibration-data` | `outputs/int8_calibration_davis_train/davis_train_calibration.yaml` | Репрезентативная calibration-выборка. |
| `--image-yolo-int8-cache-dir` | `.cache/yoloe/int8` | Кеш INT8-моделей, адресованный SHA-256 содержимого. |
| `--image-reference-sam-model` | `facebook/sam2.1-hiera-tiny` | SAM для reference foreground. |
| `--image-yolo-imgsz` | `640` | YOLOE input для кадров. |
| `--image-yolo-reference-imgsz` | `640` | YOLOE input для references. |
| `--image-yolo-confidence` | `0.10` | Detection confidence. |
| `--image-iou-threshold` | `0.30` | Ассоциация временных IDs. |
| `--image-iou-max-missed` | `1` | Допустимые пропуски track. |
| `--image-redaction` | `blur` | `blur` или `pixelate`. |
| `--image-blur-kernel-size` | `51` | Размер Gaussian blur kernel. |
| `--image-sam-mask-output-dir` | off | Каталог с PNG-масками SAM для references. |
| `--no-bboxes --no-statistics` | off | Оставить только redaction. |

### Метрики OpenVINO INT8

Бенчмарк выполнен на всех 50 кадрах DAVIS 2017 `blackswan`, 854x480, при
YOLOE input 640x640. Калибровка использовала 600 кадров из 60
последовательностей DAVIS train; `blackswan` в калибровку не входил.

| Runtime | FPS | Mean latency | p95 | J/IoU | F | J&F | Recall | Утечка | Лишний фон |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 10.42 | 95.98 ms | 126.04 ms | 0.8045 | 0.8795 | 0.8420 | 93.16% | 6.84% | 1.71% |
| OpenVINO FP32 | 5.27 | 189.68 ms | 302.25 ms | 0.8207 | 0.8977 | 0.8592 | 95.49% | 4.51% | 1.79% |
| **OpenVINO INT8** | **13.29** | **75.22 ms** | **109.35 ms** | **0.8124** | **0.8988** | **0.8556** | **94.40%** | **5.60%** | **1.77%** |

INT8 быстрее текущего PyTorch FP32 на 27.6% по FPS, а размер OpenVINO
артефакта уменьшен с 10.83 до 3.61 MiB. При этом на худшем кадре 31 утечка
достигла 49.74%, поэтому перед применением к другому реальному сценарию нужно
проверять не только средние метрики, но и покадровый worst case.

Повторяемый экспорт выполняют скрипты
`benchmarks/prepare_davis_int8_calibration.py` и
`benchmarks/export_yoloe_openvino_int8.py`.

## Offline text-prompt: Grounding DINO + SAM 2.1

Offline-режим находит заданные текстом объекты через Grounding DINO и переносит
их маски между обнаружениями через SAM 2.1:

```powershell
privacy-recognize --offline-video `
  --video-path data/videos/input.mp4 `
  --video-prompt "face" `
  --video-output data/videos/output.mp4
```

Несколько prompts задаются повторением `--video-prompt`.

## Проверка

```bash
python -m unittest discover -s tests -v
python -m compileall -q privacy_filter
```

Завершение realtime-режима: `Q`, `Esc` или `Ctrl+C`.
