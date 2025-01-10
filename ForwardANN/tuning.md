# Bayesian Optimization

```
pip install optuna
```

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    model = MyModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    train_loader = DataLoader(dataset, batch_size=batch_size)
    # Evaluate the model (return validation loss or another metric)
    performance = evaluate_model(model, train_loader)
    return performance

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
```

```shell
In-sample (Training) Performance:
total_samples: 1944
MSE: 0.4480785706298944
MAE: 0.43008927613810094
R^2: 0.9950213517959321
min_error: 1.1198892025277019e-05
max_error: 2.2606658935546875
std_error: 0.5129344842968727

Out-of-sample (Test) Performance:
total_samples: 972
MSE: 141.6827706307296
MAE: 4.974671872933288
R^2: 0.6841286794278918
min_error: 0.0003925412893295288
max_error: 52.36113357543945
std_error: 10.813667758322152
```

```shell
total_samples: 1944
MSE: 0.02348375327075816
MAE: 0.10673906002140542
R^2: 0.9997390695433573
min_error: 0.00023245811462402344
max_error: 0.8870277404785156
std_error: 0.10995692946106202

Out-of-sample (Test) Performance:
total_samples: 972
MSE: 21.317405379633215
MAE: 2.029102486402439
R^2: 0.9524744119665368
min_error: 2.008676528930664e-05
max_error: 33.255794525146484
std_error: 4.147306171397121

```


In-sample (Training) Performance:
total_samples: 4860
MSE: 0.004013520436730368
MAE: 0.03898154008966635
RMSE: 0.06335235146962083
R^2: 0.9999457181548842
min_error: 2.8875206226075534e-07
max_error: 0.521848201751709
std_error: 0.049939563163569047

Out-of-sample (Test) Performance:
total_samples: 2106
MSE: 12.324848032916915
MAE: 1.662446990289164
RMSE: 3.5106762928126702
R^2: 0.9711010857155665
min_error: 2.7104592875693e-11
max_error: 23.075820922851562
std_error: 3.092105761030081


In-sample (Training) Performance:
total_samples: 3888
MSE: 0.0012189744469742175
MAE: 0.02073877436812319
RMSE: 0.03491381455776807
R^2: 0.9999836684924746
min_error: 0.0
max_error: 0.19295692443848012
std_error: 0.028086966455676446

Out-of-sample (Test) Performance:
total_samples: 2106
MSE: 7.915297058149896
MAE: 0.9870534933270234
RMSE: 2.813413773007784
R^2: 0.9814404615287442
min_error: 0.0
max_error: 19.436080932617188
std_error: 2.6345820274686487
