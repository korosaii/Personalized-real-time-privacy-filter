# Отчёт: несколько владельцев на ChokePoint

## Что проверялось

Владельцами назначены ChokePoint `person id` `0001`, `0003` и `0006`.
Остальные 22 человека считаются посторонними и должны оставаться
пикселизированными. Использованы `yolo11-pose-roll90`,
`r50-webface600k`, порог `0.35`, пять enrollment-кадров на владельца и три
последовательных подтверждения перед открытием лица.

## Два прохода

| Метрика | Pass 1: enroll C1 → test C2 | Pass 2: enroll C2 → test C1 |
|---|---:|---:|
| GT-кадры владельцев | 144 | 193 |
| GT-кадры посторонних | 929 | 1189 |
| Владелец правильно определён среди допущенных по размеру | 100% (92/92) | 100% (87/87) |
| Владелец открыт после подтверждений, от всех его кадров | 59.03% | 41.97% |
| Кадры владельцев, остановленные gate `<80 px` | 36.11% (52/144) | 54.92% (106/193) |
| Владелец принят за другого владельца | 0% | 0% |
| Подтверждённая ложная авторизация постороннего | 0% | 0% |
| Privacy recall посторонних | 100% | 99.92% |
| Face-active pipeline p95 | 144.56 ms | 149.54 ms |
| Расчётный face-active FPS | 9.34 | 9.84 |

Сводно recognition правильно выбрал владельца на всех `179/179` кадрах,
которые прошли size gate. Полная доля открытых кадров владельцев равна 49.26%,
потому что `158/337` кадров намеренно не распознавались при bbox меньше 80 px,
а начало каждого трека скрыто до трёх подтверждений.

Privacy recall посторонних составил 99.95%. Единственный detector leak —
`P1E_S1_C1`, кадр `00004343`, person `0004`: лицо почти полностью находится
за правой границей изображения. Ни один посторонний не прошёл recognition и
не получил подтверждённую авторизацию.

Общий статус — `FAIL`: не выполнены требования открытия владельцев ≥80% и
p95 ≤33.3 ms для 30 FPS. Требования отсутствия путаницы владельцев, ложной
авторизации ≤0.1% и privacy recall ≥99% выполнены.

Performance здесь считается только на кадрах с GT-лицом и включает detector,
recognition при прохождении size gate, выбор по gallery и редактирование. Это
нагруженный face-active режим, а не средний FPS всего видео.

## Как использовался GT

Исходные JPG и XML не изменялись. Для каждого `<frame>`:

1. `number` связывает запись XML с одноимённым JPG.
2. `person id` преобразуется в ожидаемый класс: один из владельцев или
   посторонний.
3. `leftEye` и `rightEye` используются только для выбора detector bbox,
   закрывающего размеченное лицо, и контроля detector leak.
4. Embedding вычисляется recognition-моделью из найденного bbox/ландмарков.
5. Cosine score сравнивается с заранее заданным `0.35`; GT не влияет на score
   и не используется для подбора порога.

В Pass 1 templates создаются только по C1, а оценка выполняется только по C2.
В Pass 2 направление меняется. Поэтому внутри каждого прохода enrollment и
evaluation изображения не совпадают.

## Самостоятельный запуск

Полный воспроизводимый тест:

```powershell
python benchmarks\chokepoint_multi_owner_benchmark.py `
  --sequence P1E_S1 `
  --owners 0001,0003,0006 `
  --minimum-owner-face-size 80
```

Другой набор владельцев и настройки:

```powershell
python benchmarks\chokepoint_multi_owner_benchmark.py `
  --sequence P1E_S1 `
  --owners 0004,0010 `
  --enrollment-samples 7 `
  --threshold 0.38 `
  --minimum-owner-face-size 70 `
  --output-prefix benchmarks\my_multi_owner_test
```

Команда создаёт:

- агрегированный JSON и покадровый CSV с указанным `output-prefix`;
- отдельные `.npz` templates каждого владельца в
  `data/enrollments/chokepoint_multi_owner/pass_1` и `pass_2`.

Запуск обычного runtime с несколькими реальными владельцами:

```powershell
privacy-enroll alice data\photos\alice --detector yolo11-pose-roll90 --model r50-webface600k --output data\enrollments\alice.npz
privacy-enroll bob data\photos\bob --detector yolo11-pose-roll90 --model r50-webface600k --output data\enrollments\bob.npz

privacy-recognize `
  --detector yolo11-pose-roll90 `
  --model r50-webface600k `
  --minimum-owner-face-size 70 `
  --template data\enrollments\alice.npz `
  --template data\enrollments\bob.npz
```

Вместо повторения `--template` можно указать отдельный каталог, содержащий
только templates владельцев, созданные одной recognition model, с одинаковыми
detector preprocessing и режимом `--rotations`. Например:

```powershell
privacy-recognize `
  --detector yolo11-pose-roll90 `
  --model r50-webface600k `
  --minimum-owner-face-size 70 `
  --template data\enrollments\family
```
