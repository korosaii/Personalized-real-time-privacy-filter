# YOLO11n-face с пятью landmarks

Этот каталог — отдельный обучающий пайплайн. Он не меняет рабочую модель privacy-filter, пока новый detector не обучен, не проверен и не подключён отдельным изменением runtime-кода.

Модель получает стандартную голову Ultralytics `Pose`, которая за один проход выдаёт:

- bounding box лица;
- confidence класса `face`;
- левый глаз, правый глаз, нос, левый и правый угол рта;
- visibility для каждой точки.

Формат одной строки разметки:

```text
class cx cy width height x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4 x5 y5 v5
```

Все координаты нормализованы в диапазон `0..1`. `v=2` означает размеченную видимую точку, `v=0` — отсутствующую точку.

## Ускорение на MacBook M4 Pro

PyTorch обучает модель через `device=mps`, то есть на Apple GPU через Metal. Apple Neural Engine не является доступным устройством для обычного PyTorch training. После обучения модель можно экспортировать в Core ML; тогда Core ML сам распределяет inference между Neural Engine, GPU и CPU.

## 1. Окружение

Из корня репозитория:

```bash
cd training/landmarks
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверка MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

На M4 Pro результат должен быть `True`. Если он `False`, обучение не запускай: сначала проверь, что терминал использует arm64 Python, актуальные macOS и PyTorch.

## 2. WIDER FACE

Создай каталоги:

```bash
mkdir -p data/downloads data/raw
```

Скачай train, val и официальные split-аннотации. Ссылки ниже ведут на зеркало CUHK-CSE на Hugging Face, которое указано на официальной странице WIDER FACE:

```bash
curl -L -C - "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_train.zip" -o data/downloads/WIDER_train.zip
curl -L -C - "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip" -o data/downloads/WIDER_val.zip
curl -L -C - "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip" -o data/downloads/wider_face_split.zip
```

Проверка SHA-256:

```bash
shasum -a 256 data/downloads/WIDER_train.zip
shasum -a 256 data/downloads/WIDER_val.zip
shasum -a 256 data/downloads/wider_face_split.zip
```

Ожидаемые значения:

```text
e23b76129c825cafae8be944f65310b2e1ba1c76885afe732f179c41e5ed6d59  WIDER_train.zip
f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a  WIDER_val.zip
c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c  wider_face_split.zip
```

Распаковка:

```bash
unzip data/downloads/WIDER_train.zip -d data/raw
unzip data/downloads/WIDER_val.zip -d data/raw
unzip data/downloads/wider_face_split.zip -d data/raw
```

После этого должны существовать:

```text
data/raw/WIDER_train/images
data/raw/WIDER_val/images
data/raw/wider_face_split/wider_face_val_bbx_gt.txt
```

## 3. Аннотации пяти landmarks

Скачай файл `retinaface_gt_v1.1.zip` по предоставленной ссылке:

[Google Drive: retinaface_gt_v1.1.zip](https://drive.google.com/file/d/1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8/view)

Если Google Drive пишет о превышенной квоте, скачай файл позднее или через браузер с другого аккаунта. Скрипт не пытается обходить ограничение Google Drive.

Положи архив сюда:

```text
training/landmarks/data/downloads/retinaface_gt_v1.1.zip
```

Распакуй:

```bash
mkdir -p data/raw/retinaface_gt_v1.1
unzip data/downloads/retinaface_gt_v1.1.zip -d data/raw/retinaface_gt_v1.1
```

Внутри должен находиться файл `train/label.txt`. Конвертер ищет его рекурсивно, поэтому один дополнительный каталог внутри архива не мешает.

## 4. Конвертация

Старые `train2yolo.py` и `val2yolo.py` из YOLOv5-face создают другой формат. Используй адаптированные скрипты из этого каталога:

```bash
python scripts/train2yolo.py \
  --images data/raw/WIDER_train/images \
  --annotations data/raw/retinaface_gt_v1.1 \
  --output data/processed
```

Скрипт делает детерминированное разбиение train-аннотаций: 90% для обучения и 10% для pose validation. Это нужно потому, что официальная WIDER FACE val-разметка содержит bbox, но не пригодные для нашей проверки пять landmarks.

Официальную bbox validation-конвертацию запусти отдельно:

```bash
python scripts/val2yolo.py \
  --images data/raw/WIDER_val/images \
  --annotations data/raw/wider_face_split/wider_face_val_bbx_gt.txt \
  --output data/processed
```

Изображения по умолчанию не копируются: создаются ссылки на файлы из `data/raw`, поэтому датасет не занимает место второй раз. Если ссылки не подходят, добавь `--copy-images`.

Проверь формат разметки и нарисуй 12 случайных примеров:

```bash
python scripts/validate_dataset.py \
  --root data/processed/pose \
  --render 12
```

Картинки с bbox и цветными точками появятся в `artifacts/dataset_samples`. Просмотри их до обучения. Порядок точек должен быть: левый глаз, правый глаз, нос, левый угол рта, правый угол рта с точки зрения изображённого человека.

## 5. WFLW для сложных поз

WIDER FACE сохраняет качество на маленьких, далёких и групповых лицах. WFLW добавляет 10 000 лиц с 98 ручными landmarks и отдельными атрибутами `pose`, `expression`, `illumination`, `makeup`, `occlusion` и `blur`.

Скачай изображения WFLW по официальной ссылке Google Drive:

[WFLW Training and Testing Images](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)

Сохрани архив в `data/downloads`, затем скачай официальные аннотации:

```bash
curl -L -C - \
  "https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz" \
  -o data/downloads/WFLW_annotations.tar.gz
```

Распакуй оба архива так, чтобы существовали каталоги:

```text
data/raw/WFLW/WFLW_images
data/raw/WFLW/WFLW_annotations
```

Конвертация 98 точек в наши пять точек:

```bash
python scripts/wflw2yolo.py \
  --images data/raw/WFLW/WFLW_images \
  --annotations data/raw/WFLW/WFLW_annotations \
  --output data/processed
```

Конвертер создаёт:

```text
data/processed/wflw_pose
data/processed/wflw_pose_hard
data/processed/wflw_pose.yaml
data/processed/wflw_pose_hard.yaml
data/processed/widerface_wflw_train.txt
data/processed/widerface_wflw_val.txt
data/processed/widerface_wflw_pose.yaml
```

`wflw_pose_hard` содержит отдельные train/val-подмножества с атрибутом `pose`. По умолчанию его 262 train-лица включаются в manifest четыре раза, чтобы профильные ракурсы не потерялись среди обычных лиц. Файлы изображений физически не дублируются. Коэффициент можно изменить через `--hard-pose-repeats`.

После конвертации ожидается примерно 20 124 train-записи и 3 803 val-записи. В них входят WIDER FACE, все корректные WFLW-лица и усиленная hard-pose часть. Один исходный WFLW record с некорректным bbox пропускается, поэтому конвертер получает 9 999, а не 10 000 валидных face crops.

Проверь точки WFLW отдельно:

```bash
python scripts/validate_dataset.py \
  --root data/processed/wflw_pose \
  --render 30 \
  --artifacts artifacts/wflw_samples
```

## 6. Базовая проверка больших наклонов

Перед новым обучением зафиксируй качество текущего checkpoint:

```bash
python scripts/evaluate_roll.py \
  --weights runs/widerface-pose-gain12/phase2_joint/weights/best.pt \
  --dataset-root data/processed/pose \
  --split val \
  --angles=-90,-75,-60,-45,-30,0,30,45,60,75,90 \
  --limit 250 \
  --device mps \
  --output runs/roll-baseline.json
```

Скрипт отдельно считает для каждого угла:

- долю найденных лиц;
- ошибку угла линии глаз;
- ошибку пяти landmarks, нормализованную диагональю bbox;
- минимальную confidence точек.

Это важнее общего `mAP(P)` для нашей задачи: старая модель могла иметь приемлемый pose mAP, но ошибалась примерно на 53–58° при реальном повороте входа на 60°.

Зафиксируй такую же baseline на сложной WFLW pose-выборке:

```bash
python scripts/evaluate_roll.py \
  --weights runs/widerface-pose-gain12/phase2_joint/weights/best.pt \
  --dataset-root data/processed/wflw_pose_hard \
  --split val \
  --angles=-90,-75,-60,-45,-30,0,30,45,60,75,90 \
  --limit 326 \
  --device mps \
  --output runs/wflw-hard-baseline.json
```

## 7. Короткая проверка нового пайплайна

Перед полным обучением проверь один проход на 1% WFLW. Отдельный WFLW YAML здесь выбран специально: так короткий запуск гарантированно затрагивает новый датасет, а не только первые по алфавиту WIDER-файлы.

```bash
python train.py \
  --source-weights runs/widerface-pose-gain12/phase2_joint/weights/best.pt \
  --pose-data data/processed/wflw_pose.yaml \
  --device mps \
  --phase1-epochs 1 \
  --phase2-epochs 1 \
  --phase1-degrees 90 \
  --phase2-degrees 75 \
  --fraction 0.01 \
  --max-box-map50-drop 1.0 \
  --name roll90-smoke-test
```

Этот запуск проверяет код и память, но ничего не говорит о качестве модели.

## 8. Дообучение текущей Pose-модели

Запуск продолжает обучение с уже полученного `best.pt`. Landmark-голова не создаётся заново.

```bash
python train.py \
  --source-weights runs/widerface-pose-gain12/phase2_joint/weights/best.pt \
  --pose-data data/processed/widerface_wflw_pose.yaml \
  --device mps \
  --phase1-epochs 12 \
  --phase2-epochs 15 \
  --phase1-lr 0.0003 \
  --phase2-lr 0.00005 \
  --phase1-degrees 90 \
  --phase2-degrees 75 \
  --batch 8 \
  --pose-gain 12 \
  --name widerface-wflw-roll90
```

Если не хватает unified memory, уменьши только batch:

```bash
--batch 4
```

Фаза 1:

- заморожены backbone, neck, bbox и class-ветви;
- дообучается только landmark-ветвь;
- каждое изображение получает новый случайный roll в диапазоне до ±90°;
- начальный learning rate `3e-4`.

Фаза 2:

- заморожены ранние слои backbone `0..6`;
- обучаются последние слои backbone `7..10`, neck и вся Pose-голова;
- одновременно считаются `box`, `cls`, `dfl`, `pose`, `kobj` losses;
- roll ограничен ±75°, чтобы закрепить большие наклоны и стабилизировать обычные лица;
- начальный learning rate снижен до `5e-5`.

Обычная validation не содержит искусственных наклонов, поэтому её `best.pt` может выбрать раннюю эпоху, которая лучше на upright-лицах, но хуже на больших углах. По этой причине фаза 2 стартует с `phase1_landmarks/weights/last.pt`, а итоговым large-roll checkpoint считается `phase2_joint/weights/last.pt`. `best.pt` всё равно сохраняется для сравнения обычной validation.

Перед фазами исходный detector проверяется на официальном WIDER FACE validation. После второй фазы на тех же официальных изображениях проверяется box mAP50 новой модели. Отдельно на 10% landmark holdout считаются метрики пяти точек. По умолчанию разрешено падение bbox mAP50 не больше `0.03`. При большем падении скрипт возвращает код `2`, но сохраняет веса и все графики.

Результаты находятся в:

```text
runs/widerface-wflw-roll90/
├── baseline_detection
├── phase1_landmarks
├── phase2_joint
├── final_pose_validation
├── final_official_detection
└── training_summary.json
```

Главные файлы:

```text
runs/widerface-wflw-roll90/phase2_joint/weights/best.pt
runs/widerface-wflw-roll90/phase2_joint/weights/last.pt
runs/widerface-wflw-roll90/training_summary.json
```

Если первая фаза уже закончилась в старой версии скрипта, повторять её не нужно. Перезапусти только joint-фазу из правильного checkpoint:

```bash
python train.py \
  --source-weights runs/widerface-pose-gain12/phase2_joint/weights/best.pt \
  --phase1-checkpoint runs/widerface-wflw-roll90/phase1_landmarks/weights/last.pt \
  --pose-data data/processed/widerface_wflw_pose.yaml \
  --device mps \
  --phase2-epochs 15 \
  --phase2-lr 0.00005 \
  --phase2-degrees 90 \
  --batch 8 \
  --pose-gain 12 \
  --name widerface-wflw-roll90-phase2-from-last
```

## 9. Повторная угловая проверка

```bash
python scripts/evaluate_roll.py \
  --weights runs/widerface-wflw-roll90-phase2-from-last/phase2_joint/weights/last.pt \
  --dataset-root data/processed/wflw_pose_hard \
  --split val \
  --angles=-90,-75,-60,-45,-30,0,30,45,60,75,90 \
  --limit 500 \
  --device mps \
  --output runs/widerface-wflw-roll90-phase2-from-last/roll-evaluation.json
```

Сравни новый JSON с `runs/wflw-hard-baseline.json`, то есть с тем же набором изображений. `runs/roll-baseline.json` остаётся отдельной WIDER-проверкой. Угол `0°` не должен заметно ухудшиться, а median roll error на ±60° должен упасть с десятков градусов до единиц.

## 10. Подбор landmark loss

Сначала запусти `pose-gain=12`. Затем при необходимости сделай отдельные ablation-запуски, меняя только один параметр:

```bash
python train.py --source-weights runs/widerface-pose-gain12/phase2_joint/weights/best.pt --pose-data data/processed/widerface_wflw_pose.yaml --device mps --pose-gain 6 --name widerface-wflw-roll90-gain6
python train.py --source-weights runs/widerface-pose-gain12/phase2_joint/weights/best.pt --pose-data data/processed/widerface_wflw_pose.yaml --device mps --pose-gain 18 --name widerface-wflw-roll90-gain18
```

Сравнивай в `training_summary.json`:

- `metrics/mAP50(B)` и `metrics/mAP50-95(B)` — bbox;
- `metrics/mAP50(P)` и `metrics/mAP50-95(P)` — landmarks;
- `val/box_loss`, `val/cls_loss`, `val/dfl_loss` — детекция;
- `val/pose_loss`, `val/kobj_loss` — landmarks.

Не выбирай модель только по минимальному landmark loss: новый detector должен одновременно сохранить качество bbox.

## 11. Экспорт

ONNX:

```bash
python export.py \
  --weights runs/widerface-wflw-roll90-phase2-from-last/phase2_joint/weights/last.pt \
  --format onnx \
  --imgsz 640
```

Core ML для последующей проверки Apple Neural Engine:

```bash
python -m pip install coremltools
python export.py \
  --weights runs/widerface-wflw-roll90-phase2-from-last/phase2_joint/weights/last.pt \
  --format coreml \
  --imgsz 640
```

Основной runtime поддерживает Pose-выход `1×20×8400`. Экспортированную модель положи в `models/detector/yolov11n-face-pose-roll90.onnx`, затем создай новый enrollment-template и запусти сравнение с флагом `--detector yolo11-pose-roll90`. Старый `yolo11-pose` при этом остаётся отдельным baseline.

После нового enrollment проверь весь runtime, а не только координаты точек:

```bash
cd ../..
source .venv/bin/activate

privacy-enroll owner data/photos/owner \
  --detector yolo11-pose-roll90 \
  --model r34-glint360k \
  --output data/enrollments/owner-r34-pose-roll90.npz

python training/landmarks/evaluate_runtime_roll.py \
  --photos data/photos/owner \
  --template data/enrollments/owner-r34-pose-roll90.npz \
  --detector yolo11-pose-roll90 \
  --model r34-glint360k \
  --angles=-90,-75,-60,-45,-30,0,30,45,60,75,90 \
  --output training/landmarks/runs/widerface-wflw-roll90-phase2-from-last/runtime-roll.json
```

Этот тест показывает итоговый similarity и authorization recall на каждом угле. Именно он отвечает на вопрос, перестал ли privacy-filter скрывать owner при наклоне.

## Большой roll и большой yaw

`degrees=90` учит наклону головы к плечу в плоскости изображения. Это не предел в 40°: в первой фазе Ultralytics выбирает случайный угол из полного диапазона `-90..+90°`, а evaluator отдельно проверяет фиксированные углы вплоть до ±90°. Для такого roll пять точек и similarity alignment достаточны: они позволяют повернуть весь видимый рисунок лица обратно в каноническое положение.

Поворот головы в профиль — yaw — устроен иначе. Невидимую половину лица 2D similarity transform не восстанавливает. WFLW и усиленная `pose`-выборка улучшают реальные сложные ракурсы, но не дают честной гарантии для yaw около 90°. Для такого профиля нужны профильные данные, отдельная оценка по yaw-бинам и pose-aware enrollment.

Для исследовательского продолжения:

- 300W-LPA содержит yaw с шагом 5° до 90°, но требует заявку с университетской почты, разрешён только research use и запрещено распространение;
- AFLW содержит около 25 000 реальных multi-view лиц и только видимые landmarks, но также ограничен non-commercial research;
- AFLW2000-3D подходит как отдельный небольшой benchmark по yaw-бинам, но не как основной train-набор;
- LaPa содержит более 22 000 лиц и 106 точек, но его лицензия допускает только non-commercial использование;
- Menpo 3D имеет согласованную разметку frontal/profile, однако использует исходные изображения AFLW/FDDB и наследует их ограничения.

Если пять 2D-точек после этих экспериментов всё равно разваливаются на почти полном профиле, следующий технически сильный вариант — 3D face alignment вроде 3DDFA_V2. Он оценивает форму и pose головы в 3D, поэтому лучше определяет соответствия при self-occlusion. В текущий MVP его не добавляем: это ещё одна runtime-модель, отдельная интеграция и дополнительная задержка. Сначала нужно доказать на yaw-benchmark, что обученной YOLO Pose действительно недостаточно.

Практический порядок такой:

1. Обучить WIDER+WFLW checkpoint с roll до ±90° и проверить его `evaluate_roll.py`.
2. Проверить обычную детекцию на официальном WIDER FACE и не принимать модель при падении bbox mAP50 больше 0.03.
3. Для yaw проверить WFLW `pose` subset отдельно без искусственного вращения.
4. Если нужен вход в кадр сразу почти в профиль, добавить 300W-LPA или AFLW отдельным экспериментальным checkpoint и подобрать шаблоны enrollment по yaw-группам.
5. В privacy-safe runtime не авторизовывать новый трек только по экстремальному профилю: если лицо сначала было уверенно распознано фронтально, разрешение можно удерживать трекером; новый неполный профиль остаётся скрытым.

Для публичного MVP сначала используй WIDER+WFLW и roll до ±90°. Профильный checkpoint нужно хранить отдельно и не публиковать до проверки всех условий лицензий.

## Лицензии

WIDER FACE на зеркале CUHK-CSE указан как `CC BY-NC-ND 4.0`, то есть датасет нельзя автоматически считать пригодным для коммерческого обучения. На официальной странице WFLW нет ясной коммерческой лицензии, а изображения происходят из WIDER FACE, поэтому WFLW тоже считай research-only до юридической проверки. Ultralytics распространяется под `AGPL-3.0` и предлагает отдельную Enterprise-лицензию. YOLOv5-face распространяется под `GPL-3.0`; его код в этот пайплайн не копируется, ссылка используется для понимания исходного формата аннотаций. До передачи модели заказчику нужно отдельно согласовать лицензии датасета, Ultralytics и исходных весов.

Источники:

- [WIDER FACE](https://shuoyang1213.me/WIDERFACE/)
- [WIDER FACE mirror](https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data)
- [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)
- [300W-LPA](https://sites.google.com/view/300w-lpa-database)
- [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw/)
- [3DDFA / AFLW2000-3D](https://github.com/cleardusk/3DDFA)
- [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)
- [LaPa](https://github.com/jd-opensource/lapa-dataset)
- [Menpo benchmark](https://link.springer.com/article/10.1007/s11263-018-1134-y)
- [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face)
- [Ultralytics Pose dataset format](https://docs.ultralytics.com/datasets/pose/)
- [Ultralytics training on Apple Silicon](https://docs.ultralytics.com/modes/train/#apple-silicon-mps-training)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Core ML compute units](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html#specifying-compute-units)
