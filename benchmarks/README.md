# ChokePoint privacy benchmark

The dataset itself is intentionally not committed. The default `--data-root
data` layout is:

```text
data/P1E_S1/P1E_S1_C1/P1E_S1_C1/*.jpg
data/P1E_S1/P1E_S1_C2/P1E_S1_C2/*.jpg
data/P1E_S1/P1E_S1_C3/P1E_S1_C3/*.jpg
data/groundtruth/groundtruth/P1E_S1_C*.xml
```

One directory level per camera is also supported. Use `--data-root` when the
extract is stored elsewhere.

`chokepoint_privacy_benchmark.py` tests the project's face detector and
`pixelate_faces` redaction against ChokePoint eye annotations. It measures:

- exact coverage of both annotated eyes;
- consecutive privacy-leak frames;
- approximate face-zone coverage as a diagnostic;
- detector, redaction, and combined p50/p95 latency;
- a false-positive proxy on frames without eye annotations.

The default pass criteria are at least 99% privacy recall, no more than two
consecutive leak frames, and combined p95 latency within the 30 FPS frame
budget (33.3 ms). Faces with an inter-eye distance below 10 pixels are reported
but excluded from the default grade.

Run all cameras in the extracted `P1E_S1` group:

```powershell
python benchmarks\chokepoint_privacy_benchmark.py --sequence P1E_S1
```

Несколько групп можно передать повторением флага или через запятую:

```powershell
python benchmarks\chokepoint_privacy_benchmark.py `
  --sequence P1E_S1 `
  --sequence P1L_S1

python benchmarks\chokepoint_privacy_benchmark.py `
  --sequence P1E_S1,P1L_S1
```

Run a quick smoke test:

```powershell
python benchmarks\chokepoint_privacy_benchmark.py `
  --sequence P1E_S1 `
  --max-frames 350 `
  --output-prefix benchmarks\chokepoint_smoke
```

Сравнить обработку без трекера и с одним из runtime-трекеров:

```powershell
python benchmarks\chokepoint_privacy_benchmark.py --tracker none
python benchmarks\chokepoint_privacy_benchmark.py --tracker iou
python benchmarks\chokepoint_privacy_benchmark.py --tracker bytetrack
python benchmarks\chokepoint_privacy_benchmark.py --tracker botsort
```

По умолчанию используется `--tracker none`, поэтому прежний покадровый
протокол и его результаты сохраняются. При включённом трекере его время входит
в `tracker_ms` и общую latency `pipeline_ms`.

The command writes a summary JSON and a frame-level CSV below `benchmarks`.
When leaks exist, it also creates a JPG montage of the worst frames. Exit code
zero means every configured check passed; exit code one means at least one
check failed.

ChokePoint has eye coordinates, not face segmentation masks. Therefore exact
eye coverage is the default privacy criterion. The eye-derived face-zone score
is deliberately diagnostic. To enforce that heuristic too, set, for example,
`--minimum-zone-coverage 0.8`.

This benchmark exercises anonymization of all annotated faces. It does not
test owner enrollment or identity authorization.

Recorded full `P1E_S1` run (`6876` frames, DirectML): privacy recall `100%`,
zero leak frames, pipeline p95 `48.46 ms`. Privacy and temporal checks passed;
the overall result was `FAIL` only because p95 exceeded the `33.3 ms` budget
for 30 FPS.

## Тест нескольких владельцев: два прохода

`chokepoint_multi_owner_benchmark.py` назначает несколько ChokePoint `person
id` владельцами, а остальных людей считает посторонними. По умолчанию выбраны
`0001`, `0003` и `0006`.

Протокол не использует один и тот же ракурс для регистрации и проверки:

1. `pass_1`: пять кадров каждого владельца с камеры C1 создают templates;
   метрики считаются на всех размеченных лицах C2.
2. `pass_2`: templates строятся заново по C2; метрики считаются на C1.

Порог `0.35` фиксируется до evaluation. Для имитации runtime лицо открывается
только после трёх последовательных совпадений с одним и тем же владельцем.

GT из XML не превращается в новые bounding boxes и исходные JPG не меняются:

- `frame number` связывает XML с JPG;
- `person id` задаёт ожидаемую личность и класс владелец/посторонний;
- `leftEye` и `rightEye` выбирают detection, который действительно относится
  к размеченному лицу;
- GT не участвует в вычислении embedding, cosine score или подборе порога.

Полный запуск двух проходов:

```powershell
python benchmarks\chokepoint_multi_owner_benchmark.py `
  --sequence P1E_S1 `
  --owners 0001,0003,0006 `
  --minimum-owner-face-size 80
```

Тот же тест с реальной привязкой подтверждений к track ID:

```powershell
python benchmarks\chokepoint_multi_owner_benchmark.py `
  --sequence P1E_S1 `
  --tracker bytetrack
```

Поддерживаются `none`, `iou`, `bytetrack` и `botsort`. По умолчанию выбран
`none`, чтобы старые результаты оставались воспроизводимыми. При включённом
трекере его время записывается отдельно в `tracker_ms` и входит в
`face_active_pipeline_ms`.

В multi-owner тесте также можно выбрать несколько групп обоими форматами
`--sequence`. Для каждой группы templates C1/C2 строятся независимо, после
чего результаты всех проходов объединяются в общую секцию `combined`:

```powershell
python benchmarks\chokepoint_multi_owner_benchmark.py `
  --sequence P1E_S1,P1L_S1 `
  --owners 0001,0003,0006
```

Свои ID, число enrollment-кадров и порог:

```powershell
python benchmarks\chokepoint_multi_owner_benchmark.py `
  --sequence P1E_S1 `
  --owners 0004,0010 `
  --enrollment-samples 7 `
  --threshold 0.38 `
  --minimum-owner-face-size 70 `
  --output-prefix benchmarks\my_multi_owner_test
```

Результаты сохраняются в `benchmarks/chokepoint_multi_owner.json` и покадровый
`CSV`. Сгенерированные templates двух проходов находятся в
`data/enrollments/chokepoint_multi_owner/pass_1` и `pass_2`.

Основные метрики:

- `owner_raw_identification_recall` — правильный владелец до подтверждений;
- `owner_raw_identification_recall_when_attempted` — точность владельца только
  на кадрах, прошедших минимальный размер bbox;
- `owner_size_gated_rate` — доля кадров владельцев, намеренно оставшихся
  скрытыми из-за порога размера;
- `owner_confirmed_reveal_recall` — доля кадров владельцев, реально открытых
  после трёх подтверждений;
- `owner_wrong_identity_rate` — владелец принят за другого владельца;
- `stranger_confirmed_false_authorization_rate` — посторонний ошибочно открыт;
- `stranger_privacy_recall` — посторонний обнаружен и остался скрытым;
- `face_active_pipeline_ms` — detector + опциональный tracker + recognition +
  выбор владельца + редактирование на кадрах с лицом.

Тест проверяет классификацию и решение о редактировании на каждом GT-кадре.
При `--tracker none` ассоциация движения не воспроизводится. В остальных
режимах подтверждения привязываются к track ID, но runtime-расписание повторных
проверок `UNKNOWN` всё ещё не воспроизводится.

Фактические результаты текущего полного запуска и их интерпретация находятся
в `benchmarks/chokepoint_multi_owner_report.md`.

Все параметры обоих benchmark доступны через:

```powershell
python benchmarks\chokepoint_privacy_benchmark.py --help
python benchmarks\chokepoint_multi_owner_benchmark.py --help
```

JSON, CSV и failure montage считаются генерируемыми локальными артефактами и
исключены из Git. В репозитории сохраняются benchmark-код и сводный Markdown
multi-owner отчёт.
