# Personalized real-time privacy filter

Локальный фильтр для macOS Apple Silicon: зарегистрированный владелец показывается без размытия, остальные обнаруженные лица размываются.

## Требования

- macOS на Apple Silicon;
- Python 3.12;
- SCRFD-10G_KPS ONNX;
- iResNet-50@WebFace600K ONNX.

## Установка

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt
```

Разрешите доступ к камере для VS Code или Terminal в **System Settings → Privacy & Security → Camera**.

## Модели

Для запуска нужны две статические модели:

```text
models/
├── detector/
│   └── det_10g_512.onnx
└── recognition/
    └── webface_r50_112.onnx
```

`det_10g.onnx` и `webface_r50.onnx` — необязательные исходные ONNX. Если есть только они, статические модели можно создать командами:

```bash
python scripts/make_static_scrfd.py \
  models/detector/det_10g.onnx \
  models/detector/det_10g_512.onnx \
  --size 512

python scripts/make_static_recognition.py \
  models/recognition/webface_r50.onnx \
  models/recognition/webface_r50_112.onnx
```

## Регистрация владельца

Добавьте фотографии владельца в `data/photos/owner/`, затем выполните:

```bash
privacy-enroll owner data/photos/owner
```

Шаблон будет сохранён локально в `data/enrollments/owner.npz`. Фотографии и биометрический шаблон исключены из Git.

## Запуск

```bash
privacy-recognize
```

Завершение: `Q`, `Esc` или `Ctrl+C`.

Полезные команды:

```bash
privacy-recognize --help
privacy-enroll --help
inspect-models --help
```

## Локальные файлы

ONNX weights, фотографии, enrollment templates, benchmarks, caches и виртуальное окружение не публикуются в GitHub.
