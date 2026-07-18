# Solver v3 — консолидированный солвер (итоговый мердж)

База: solver-v3 @ `ad44753`.
Поглощает Stage-A ветку разработки (закрывается без мерджа) и делает репозиторий
готовым к удалению всех промежуточных веток после мерджа в `main`.

## Что вошло

1. **База агента (packed group-scale counter + v3-цикл)** — без изменений поведения
   по умолчанию: `gptq_group_ternary(..., refine_iters=2, scale_refit="hdiag")`.
2. **A5: ITF-сетка** (`grid="itf"`): асимметричная тернарная сетка {−s_neg, 0, +s_pos}
   на группу, координатный спуск (переназначение ↔ per-lobe L2 refit), инициализация
   точным per-lobe `optimal_ternary`. Упаковка в packed-счётчик: поддержка от itf
   фиксируется, sym-масштаб точно решается заново (align) — это строго лучше
   усреднения долей.
3. **A7: align-refit** (`scale_refit="align"`): точный совместный пересчёт масштабов
   строки в H-метрике на фиксированной поддержке (K базисов на строку, один
   `torch.linalg.solve` K×K, кламп положительности). Заменяет greedy `hessian_cd`:
   точный совместный решающий ≥ greedy координатного спуска по построению, дешевле.
4. **A4.1: salient-first** (`salient_first>0`): доля каналов с топ |w|·√diag(H)
   выносится ДО sweep в компоненту s2·sign(w) (BiLLM-стиль), t=0 на них, s2 =
   per-(row,group) mean|w| по salient; внутри error feedback.
5. **Расширение packed-формата salient-каналом** (оба counter-слоя):
   `salient_idx` (int32, flat original-order) + `salient_val` (fp16) — точные значения
   salient-весов поверх 6-битного состояния; базовое (t,c) обнуляется на salient и
   **заморожено через все апдейты** (reference-путь явно перезаписывает коды
   `_salient_zero_code`; Triton strict-update отключается при наличии salient —
   на калибровочных batch reference-путь быстрее и так). В forward/grad-x Triton
   добавляется разреженная COO-коррекция (кэшируется, rebuild только в
   `load_group_state`). Учёт в `persistent_bytes`/`state_statistics`. Стоимость:
   6 байт на salient-вес (≈ +0.48 bpw при 1%).
6. **Свёртка review_fixes в исходники** — `src/memory_native/review_fixes.py`
   УДАЛЁН, monkey-patch слой ликвидирован:
   - `_carry_resolve` — каноническая реализация из `counter.py` (с `strict=False`);
   - power-of-two проверка group в `triton_group_counter_update_from_io` и в
     `PackedGroupScaleCounterLinear` (strict Triton update требует pow2 group);
   - `sr_step` — персистентный буфер (попадает в state_dict), зеркалирует `_sr_step`;
   - flip-sample телеметрия: `flip_sample_size`, `_sample_codes`,
     `observe_flip_sample`, `flip_rate_alt`, `counter_edge_sample` (low |c|),
     auto-hook в `_sync_after_load`;
   - `ptq_warm_start` фильтрует counter-only kwargs для reference-пути.

## Исключено (измеренно)

- **A6 / SSR** (приоритет знака по |H_offdiag|): +94% к ошибке слоя на smoke,
  +14% на layerwise — вреден, не включён.

## Gate (layerwise, измерено)

Скрипт `scripts/run_solver_v3_consolidated_layerwise.py`, калибровка
`scripts/solver_v3_calib_probe.py`. Результаты: `results/solver_v3_consolidated_layerwise.{json,md}`.
Конфиг: Qwen2.5-0.5B, слои [0, 23], WikiText-2, 8×[2,128] seed 0, CPU fp32.

| arm | rel H-err | vs base |
|---|---|---|
| v3_base | 0.01765 | — |
| v3_hesscd | 0.01511 | −14.4% |
| v3_align | 0.01485 | −15.9% |
| v3_itf | 0.01457 | −17.5% |
| v3_salient | 0.01604 | −9.1% |
| **v3_full** (itf+align+salient) | **0.01371** | **−22.3%** |
| v3_full, refine_iters=4 | 0.01349 | −23.6% |
| v3_full, refine_iters=6 | 0.01344 | −23.8% |

Ингредиенты аддитивны и совместимы с v3-циклом агента. Align ≥ hessian_cd
подтверждено на реальных данных (0.01485 < 0.01511).

Честная оговорка: v3-цикл агента сам по себе стартует слабее старой ветки
(0.01765 против 0.01437 v2), и цепочка (0.01371) пока чуть выше старой цепочки
(0.01292, соло-настройки). Разрыв — отсутствие in-sweep per-group refit'а в общем
коде (была ликвидирована регрессия GPTQ-свипа). Следующий рычаг: вернуть in-sweep
refit как опцию (`refit="group"`), ожидаемый эффект ~5-6% по истории веток.

## Тесты

`tests/test_solver_v3_consolidated.py` — 13 тестов:
ITF (выигрыш на skewed, монотонность, ловушка sym-инициализации), align
(standalone ≤ hdiag и через solver API), salient_first (выигрыш на heavy-tail,
точное разложение, восстановление), packed-формат (roundtrip, sparse-коррекция
== dense fwd/grad, заморозка через апдейты, байты/статистика), reference-слой,
`ptq_warm_start` end-to-end (включая itf→pack через align). Вся репо-суита зелёная.

## Формат: обратная совместимость

- API по умолчанию не меняется: `grid="sym"`, `salient_first=0.0`,
  `refine_iters=2`, `scale_refit="hdiag"`.
- `load_group_state(scales, t, c, perm)` без salient-параметров — прежнее поведение.
- Старые state_dict без salient-буферов грузятся (буферы пустые по умолчанию).
