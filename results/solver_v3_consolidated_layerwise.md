# Solver v3 CONSOLIDATED — layerwise gate

Date: 2026-07-17.
Donor: Qwen/Qwen2.5-0.5B, bf16; calib WikiText-2 train 8×[2,128]; sampled layers 0+23
(14 linears); metric: relative H-weighted layer output error Σ(w−q)ᵀH(w−q)/ΣwᵀHw
(the exact objective the solver optimizes). NO training anywhere. bpw ≈ 1.71 +
сальентный канал (+0.48 при salient_first=0.01).

| arm | rel_err | vs v3_base | сек |
|---|---|---|---|
| v3_base (agent default: refine 2 × hdiag) | 0.01765 | — | 40 |
| v3_hesscd (greedy H-CD refit) | 0.01511 | −14.4% | 131 |
| v3_align (A7 exact joint solve) | 0.01485 | −15.9% | 62 |
| v3_itf (A5 + align) | 0.01457 | −17.5% | 75 |
| v3_salient (A4.1, 1%) | 0.01604 | −9.1% | 47 |
| **v3_full (A5 + A7 + A4.1)** | **0.01371** | **−22.3%** | 86 |
| v3_full, refine_iters=4 | 0.01349 | −23.6% | 174 |
| v3_full, refine_iters=6 | 0.01344 | −23.8% | 183 |

## Выводы

1. Ингредиенты Stage-A переносятся на v3-цикл как задумано и аддитивны: точный align
   бьёт жадный hessian_cd на той же базе (0.01485 < 0.01511 — подтверждено
   «exact ≥ greedy на том же саппорте» на реальных данных); itf и salient_first
   докладывают каждый своё; полная цепочка — лучшая: −22.3% относительно базы ветки.
2. Честный сюрприз: сам v3-цикл agent-ветки (0.01765) стартует ХУЖЕ старого v2
   (0.01437, та же методика на finetune-ветке), а полная консолидированная цепочка
   (0.01371) чуть не дотягивает до finetune-цепочки A5+A7+A4.1 (0.01292). Прирост
   refine_iters (4/6) почти не помогает — разрыв в КАЧЕСТВЕ СТАРТА цикла: у agent-свипа
   нет внутрисвипового per-group refit, который был у v2 (refine_scale=True).
   Следующий рычаг (не входит в этот merge): вернуть внутрисвиповый refit как опцию
   цикла — ортогонально ингредиентам; ожидаемый эффект по аналогии с обеими базами
   ≈ 5-6% и закрытие разрыва.
3. Packed-видимый вес при grid="itf" получается точным sym-пересчётом (align) на
   itf-саппорте; сальент хранится точным fp16-каналом. Юнит-тождества закреплены в
   tests/test_solver_v3_consolidated.py.

Raw: results/solver_v3_consolidated_layerwise.json. Протокол воспроизводим:
scripts/solver_v3_calib_probe.py (субпроцесс) + scripts/run_solver_v3_consolidated_layerwise.py.
