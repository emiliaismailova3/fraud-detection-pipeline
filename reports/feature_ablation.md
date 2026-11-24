LightGBM, same time split (train < day 122, valid < day 152, test after), seed 42.

| feature set            |   features |   pr_auc |   roc_auc |   recall_at_1%_alerts |   precision_at_1%_alerts |
|:-----------------------|-----------:|---------:|----------:|----------------------:|-------------------------:|
| base (served model)    |         45 |   0.5208 |    0.9008 |                0.2434 |                   0.8479 |
| + C/D/M                |         66 |   0.5248 |    0.9013 |                0.2428 |                   0.8456 |
| + C/D/M + V            |        405 |   0.5458 |    0.9038 |                0.255  |                   0.8881 |
| + C/D/M + V + identity |        445 |   0.541  |    0.9033 |                0.2527 |                   0.8803 |
