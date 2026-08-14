# Personalized real-time privacy filter

Real-time фильтр для камеры: зарегистрированный владелец остаётся видимым, остальные обнаруженные лица скрываются пикселизацией.

Проект использует YOLO11n-face, MobileFaceNet@WebFace600K, ONNX Runtime и OpenCV. Лендмарки и дополнительные landmark-модели не используются. На Apple Silicon автоматически используется CoreML. На Windows автоматически используется DirectML с совместимой NVIDIA, AMD или Intel GPU, а при недоступности GPU — CPU.

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
    └── w600k_mbf.onnx
```

Runtime использует статический YOLO11 ONNX с входом `640×640` и MobileFaceNet с входом `112×112`.


## Регистрация владельца

Добавьте одну или несколько фотографий в:

```text
data/photos/owner/
```

Команда на macOS и Windows:

```bash
privacy-enroll owner data/photos/owner
```

Biometric template сохраняется в `data/enrollments/owner.npz`. Исходные фотографии и enrollment data исключены из Git.

Порог авторизации по умолчанию: `0.35`.

## Запуск

```bash
privacy-recognize
```

Лица с bbox меньше `80 px` остаются скрытыми и не отправляются в recognition. Перед первой проверкой лицо должно полностью находиться внутри кадра, а трек — быть стабильным три кадра. Новый подходящий трек проходит три проверки. После авторизации стабильный трек сохраняет состояние без постоянного recognition, в том числе при частичном выходе за край кадра. UNKNOWN проверяется повторно после роста на 15%, заметного перемещения в кадре или потери уверенности трекера. Авторизация сохраняется при уменьшении лица до `56 px`.

Завершение: `Q`, `Esc` или `Ctrl+C`. Последний benchmark сохраняется локально в `benchmarks/latest.json`.

Дополнительные параметры:

```bash
privacy-recognize --help
privacy-enroll --help
inspect-models --help
```
