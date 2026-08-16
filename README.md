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
│   └── yolov11n-face.onnx
└── recognition/
    └── glintr100.onnx
```

Runtime использует статический YOLO11 ONNX с входом `640×640` и IResNet100 с
входом `112×112` и выходом `512`.

SHA-256 `glintr100.onnx`:
`4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf`.
Предобученные модели InsightFace предоставляются только для некоммерческих
исследовательских целей; для коммерческого использования нужна отдельная
лицензия или собственные веса.


## Регистрация владельца

Добавьте одну или несколько фотографий в:

```text
data/photos/owner/
```

Команда на macOS и Windows:

```bash
privacy-enroll owner data/photos/owner
```

После смены recognition-модели существующий biometric template необходимо
пересчитать той же командой. Новый `owner.npz` атомарно заменит старый шаблон.
Индексы rotation-centroid фиксированы порядком `0=0°`, `1=90°`, `2=180°`,
`3=270°`. После recognition рядом с bbox выводятся максимальная косинусная
близость, индекс победившего центроида `IDX` и его угол `R`.

Biometric template сохраняется в `data/enrollments/owner.npz`. Исходные фотографии и enrollment data исключены из Git.

Порог авторизации по умолчанию: `0.35`.

## Запуск

```bash
privacy-recognize
```

Изображение камеры зеркалится по умолчанию. Для обычного незеркального отображения:

```bash
privacy-recognize --no-mirror
```

Лица с bbox меньше `80 px` остаются скрытыми и не отправляются в recognition. Перед первой проверкой лицо должно полностью находиться внутри кадра, а трек — быть стабильным три кадра. Новый подходящий трек проходит три проверки. После авторизации стабильный трек сохраняет состояние без постоянного recognition, в том числе при частичном выходе за край кадра. UNKNOWN проверяется повторно после роста на 15%, заметного перемещения в кадре или потери уверенности трекера. Авторизация сохраняется при уменьшении лица до `56 px`.

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
