# TabM-MLX

Benchmarks on datasets from [TabReD](https://github.com/yandex-research/tabred) benchmark. Numbers display throughput in batches per second with batch size equal to 1 on M3 macbook.

|       dataset / model      | catboost | xgboost | lightgbm | lleaves | TabM-mini | TabM-mini + embeddings | TabM-mini torch | TabM-mini + embeddings torch |
|:--------------------------:|----------|---------|----------|---------|-----------|------------------------|-----------------|------------------------------|
| california (8 num)         | 6852     | 11148   | 5157     | 22226   | 44536     | 17662                  | 9097            | 6100                         |
| diamond (6 num, 3 cat)     | 7785     | 11199   | 5421     | 21282   | 18861     | 11813                  | -               | -                            |
| adult (6 num, 8 cat)       | 14578    | 9972    | 4127     | 20895   | 9807      | 7532                   | -               | -                            |
| cook-time (186 num, 6 cat) | 5225     | 11363   | 3871     | 20803   | 11592     | 8281                   | -               | -                            |
| weather (100 num, 3 cat)   | 6188     | 11158   | 4561     | 21574   | 17331     | 10208                  | -               | -                            |
