# Personalized real-time privacy filter

Real-time фильтр для камеры: зарегистрированный владелец остаётся видимым, остальные обнаруженные лица размываются.

Проект использует SCRFD-10G_KPS, iResNet-50/ArcFace, ONNX Runtime и OpenCV. Он работает на macOS и Windows. На Apple Silicon автоматически используется CoreML. На Windows автоматически используется DirectML с любой совместимой NVIDIA, AMD или Intel GPU, а при недоступности GPU — CPU.

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
│   └── det_10g.onnx
└── recognition/
    └── webface_r50.onnx
```

Вручную преобразовывать модели не нужно. При первой команде программа автоматически создаст копии с фиксированными shapes внутри `.cache/models/`.


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

## Запуск

```bash
privacy-recognize
```

Завершение: `Q`, `Esc` или `Ctrl+C`. Последний benchmark сохраняется локально в `benchmarks/latest.json`.

Дополнительные параметры:

```bash
privacy-recognize --help
privacy-enroll --help
inspect-models --help
```
