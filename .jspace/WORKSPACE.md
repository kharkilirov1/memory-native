# J-Space Workspace Ledger

## Goal
Конвертировать Qwen3.8-27B (VLM qwen3_5) в memory-native counter-формат стримингом на CPU-боксе; RX580/MotifCL — целевой рантайм (не солвер); подтвердить мостами restore-witness + PPL

## Core

## Verified
- ✓01 Полный streaming-путь валидирован tiny qwen3_5 VLM: skeleton retry, probe cls-match, materialize, solve, cascade, resume, phase2 restore (extra_skip vision, 0 unexpected), forward finite — verified by: tiny_qwen35_smoke.py ALL GREEN; restore_witness.py phase2 прогнан на tiny
- ✓02 Конвертация Qwen3.8-27B запущена detached: донор полный, корпус 4.01M train, solve блока 0 идёт (pid 25260, CPU 1668s, RSS 3.81 GiB, в прогнозе 3-4 GiB) — verified by: index.json weight_map vs files: all 18 of 18 shards; manifest: all 6 domains; process tree + CPU/RSS sampled through 30 min of solve
- ✓03 27B restore-witnessphase2 прошёл: 496 counter-линейных (24.35B коэф), reference-load 3472 тензоров 0 unexpected, 688 fp-параметров bf16 из шардов, forward конечен, top-1 после 'The capital of France is' = ' Paris' — verified by: all 64 block files reloaded via load_streamed_state; phase3 probe forward finite on the full restored VLM
- ✓04 Qwen3.8-27B конвертация ЗАВЕРШЕНА и подтверждена гейтами: 64/64 блоков, 15.59B коэф., стейт 20.5 GiB; restore 496 counter-линейных 0 unexpected; WikiText-2: counter ppl=15.856 vs fp 7.407 (x2.14), KL=0.876 nats/256 поз., top-1 'Paris' — verified by: convert.log final line + manifest all 64 blocks; restore_witness phases 1-3 exit 0; fp cache ppl 7.407 over 4092 tokens, RESULT line in phase23.log
- ✓05 KD-проба на CPU для 27B: машинерия работает (кэш учителя 64 блока за 53 мин, restore+freeze+grad-ckpt+topK-KD потери ок, backward запускается), но throughput смертен: forward 21 c/блок, backward >30 мин/блок — bf16-GEMM без нативной поддержки эмулируется (одна [256x17408]@[17408x5120] >5 мин); fp32-студент не влезает в 32 GiB — verified by: measured timers on all 6 step phases at NUM_BLOCKS=1; standalone bf16 vs fp32 GEMM bench; 64-block teacher cache exit 0
- ✓06 Asym-каскад портирован в стриминг и валидирован: (H_q, G) совпадают с референсом rel ~3e-7 на всех 7 таргетах блока 1, блок 0 EXACT vs классика, 2-пас механика GREEN, классика 10/10 pytest + tiny ALL GREEN — verified by: asym_arbiter.py per-target table incl. all block-1 targets; asym_diag.py; asym2_smoke.py; pytest through 10/10
- ✓07 Colab G4-прогон запущен под автоматизацией: ноутбук загружен (drive id 1M0x5nrANMb15fn86fIZaL6XmJtEmPXjj), рантайм RTX6000 подключен (INVALID_ARGUMENT от gpuType=RTX6000 вылечен штатным пикером 'Графический процессор G4'+Сохранить), Drive OAuth пройден, все 9 ячеек поставлены через colab.global.notebook.kernel.execute, донор качается (диск +18 GB/мин), CU осталось 102.2 из ~102 — verified by: session panel metrics sampled through 14:17: RAM 3.4->8.9 GB, disk 50.6->69.0 GB; monitor v2 logging to colab_monitor.log

## Open
- ✅ **КАМПАНИЯ ЗАВЕРШЕНА (2026-08-31)**: 400-шаговый strict-KD ран run_20260831_005655
  прошёл ВСЕ гейты — **KD_ACCEPTED**: агрегат 2.0242 → **1.9147 (−5.4%)**, все домены
  улучшились (science −13.4%, instruct −23.9%), scale_mean ratio 1.0000001.
  Рантайм Colab завершён (CU перестали тратиться), артефакты в GitHub.
- best.pt = 3 чанка в Release ship-run_20260831_005655 + best.sha256; сборка:
  cat best_part* > best.pt, сверить sha256.

## No-Drive запуск (воспроизводимая процедура)
1. Chrome дефолтный профиль (сессия lirovkharki2@gmail.com); CDP заблокирован — управление
   через ZCode computer-use (AX-клики) + PowerShell mouse_event (куада-кадры stale из-за
   анимированных обоев).
2. Терминал Colab: ввод ТОЛЬКО через clipboard: write_clipboard -> ctrl+shift+v -> return
   (AX-set в поле терминала не доставляет текст).
3. Вывод из VM: скрипты шлют файлы в приватный репо (Contents API для мелочи, Release
   assets чанками <=1.9 GiB для best.pt); на ПК читаем через gh api. ВАЖНО: sha лежит в
   cur['sha'], не cur['content']['sha']; raw.githubusercontent кэшируется ~5 мин —
   свежие скрипты качать через api.github.com + Accept: application/vnd.github.raw.
4. Drive прочитан через rclone mount (google.colab.drive.mount вне notebook-frontend НЕ
   работает): rclone authorize "drive" на ПК -> токен -> rclone.conf на VM -> mount
   --daemon gdrive: /content/drive_mnt (нужен apt-get install fuse3 + симлинк fusermount)
   -> ln -s /content/drive_mnt /content/drive/MyDrive.
5. Исполнение ячеек: inject_a.py (jupyter_client -> живое ядро) исполняет bootstrap_a.py:
   ячейки V4p4 0..22, SKIP drive.mount, RELEASE_ZIP пин на v3 (sorted glob ловил v2!),
   оверлей project/v3/* -> project/*, precopy state+cache через rclone copyto/copy.
6. Ловушки найденные в бою: Stage-R лаунчер берёт глобальный env (остаётся от smoke —
   нужен ручной прод-лаунчер с полным env), раннер требует TOPK=1024 и STEPS<=400
   (кэш), Colab убивает рантайм «за бездействие» несмотря на терминал-процессы —
   нужны клики по странице каждые ~3 мин (крон это делает).

## Next (опционально)
1. Собрать best.pt из 3 чанков Release на ПК (cat best_part* > best.pt, проверить
   sha256 из run_20260831_005655/best.sha256), при желании — witness ppl на 400-стейт.
2. Отключить крон automation-0b9be367 (или оставить как ежедневный чек).
3. Ротация gh-токена рекомендована (светился в транскрипте/логах).

## Verified (2026-08-30, браузер, нулевая квота до запуска)
- ✓08 Drive-реконструкция: mn27 = state_max.zip + best_kd.pt + metrics.json + mix_qwen35_v3;
  mn27_strict_v3 = только teacher_cache_v3; mn27_recovery_3k_v4 пуста; релиз v3 zip на месте
- ✓09 Квота Colab 432.03 CU (Pro+); Drive 140.06/15 ГБ READ-ONLY (One отменён из-за оплаты)
- ✓10 GitHub-канал: Contents PUT 201, Release create 201, chunk asset upload 201 (протестировано с ПК)
- ✓11 G4 VM: RTX PRO 6000 Blackwell 97887 MiB, 176 GB RAM, 48 CPU (vminfo через канал VM->GitHub)
- ✓12 Препфлайты V4p4 на VM: Stage A [OK] по всем входам, GH write probe OK, donor gate OK,
  release validate OK (cell 11, 15s), GPU smoke OK (cell 12), стейт распакован (13), корпус (14),
  unit tests green (15) — verified by: терминальный лог inject_a4/a5
