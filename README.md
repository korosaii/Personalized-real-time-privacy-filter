# Personalized real-time privacy filter

Real-time фильтр для камеры: зарегистрированный владелец показывается без размытия, остальные обнаруженные лица размываются.

Проект оптимизирован для macOS Apple Silicon и использует SCRFD, iResNet-50/ArcFace, ONNX Runtime, CoreML и OpenCV.

## Клонирование и установка

```bash
git clone https://github.com/korosaii/Personalized-real-time-privacy-filter.git
cd Personalized-real-time-privacy-filter

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt
```

На macOS разрешите доступ к камере для VS Code или Terminal в **System Settings → Privacy & Security → Camera** и перезапустите приложение.

## Модели

Веса моделей не входят в репозиторий. Получайте pretrained models из официальных источников InsightFace и соблюдайте их условия использования.

Для запуска нужны два локальных файла:

```text
models/
├── detector/
│   └── det_10g_512.onnx
└── recognition/
    └── webface_r50_112.onnx
```

Runtime использует:

- SCRFD-10G_KPS с фиксированным входом `1×3×512×512`;
- iResNet-50@WebFace600K с фиксированным входом `1×3×112×112` и выходом `1×512`.

Если файлы `_512.onnx` и `_112.onnx` уже есть, следующий раздел можно пропустить.

## Подготовка статических ONNX

Исходные ONNX могут иметь динамические dimensions. CoreML стабильнее компилирует графы с фиксированными shapes, поэтому проект содержит два одноразовых вспомогательных скрипта.

Ожидаемые исходные файлы:

```text
models/detector/det_10g.onnx
models/recognition/webface_r50.onnx
```

Создание runtime detector:

```bash
python scripts/make_static_scrfd.py \
  models/detector/det_10g.onnx \
  models/detector/det_10g_512.onnx \
  --size 512
```

Аргументы означают:

1. путь к исходному SCRFD ONNX;
2. путь нового статического ONNX;
3. фиксированный размер detector input.

Создание runtime recognition-модели:

```bash
python scripts/make_static_recognition.py \
  models/recognition/webface_r50.onnx \
  models/recognition/webface_r50_112.onnx
```

Аргументы означают:

1. путь к исходному iResNet-50 ONNX;
2. путь нового ONNX с фиксированными shapes `1×3×112×112 → 1×512`.

Скрипты не обучают, не квантуют и не меняют веса моделей. Они создают локальные копии графов с фиксированными tensor shapes для CoreML.

Проверка подготовленных файлов:

```bash
cd models
shasum -a 256 -c MANIFEST.sha256
cd ..
```

## Регистрация владельца

Добавьте одну или несколько фотографий в:

```text
data/photos/owner/
```

Создайте локальный biometric template:

```bash
privacy-enroll owner data/photos/owner
```

Результат сохраняется в `data/enrollments/owner.npz`. Фотографии и enrollment template исключены из Git.

## Запуск

```bash
privacy-recognize
```

Завершение: `Q`, `Esc` или `Ctrl+C`.

Benchmark сохраняется локально в `benchmarks/latest.json` и не публикуется в Git.

Дополнительные параметры:

```bash
privacy-recognize --help
privacy-enroll --help
inspect-models --help
```

## Лицензирование моделей

Исходный код InsightFace опубликован под MIT License, но предоставляемые InsightFace training data и pretrained models ограничены non-commercial research use. Это ограничение относится и к моделям, скачанным вручную, и к моделям, загруженным средствами библиотеки.

Этот репозиторий не распространяет ONNX/PTH weights. Статическая копия ONNX сохраняет исходные веса и не меняет их лицензионные условия. Для коммерческого применения или публичного перераспространения weights заранее получите подходящее разрешение у правообладателя.

- InsightFace license: <https://github.com/deepinsight/insightface#license>
- InsightFace model zoo: <https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo>

## Ограничения

- detector теоретически может пропустить лицо;
- liveness detection пока отсутствует;
- threshold требует проверки на целевых условиях;
- multi-face tracking нужно дополнительно валидировать на реальных людях;
- это исследовательский privacy-прототип, а не система биометрического контроля доступа.
