# Özellik sözlüğü (K2 doldurur)

| Sütun deseni | Formül | Birim | Klinik anlam |
|---|---|---|---|
| `<kanal>__mean` | μ = (1/L) Σ xᵢ | kanal birimi | |
| `<kanal>__std` | ddof=1 | | |
| `<kanal>__var` | σ² | | |
| `<kanal>__rms` | √((1/L) Σ xᵢ²) | | |
| `<kanal>__cov` | 100·σ/μ | % | Adım-adım değişkenlik; HD ve ALS'de artar |
| `<kanal>__dominant_freq` | argmax_{k≥1} P_k / L | döngü/adım | |
| `<kanal>__spectral_entropy` | −Σ p log₂ p / log₂ K | 0–1 | |
| `asym__<sol>__<sağ>` | 100·\|μ_L−μ_R\| / (0,5(μ_L+μ_R)) | % | Sol-sağ asimetri; PD'de tek taraflı başlangıç |

Kanallar: stride, swing, stance (s ve %), double support (s ve %) – bkz. `config/config.yaml`.
