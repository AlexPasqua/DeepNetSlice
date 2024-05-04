# DeepNetSlice
### _A Deep Reinforcement Learning Open-Source Toolkit for Network Slice Placement_

## Demo
```bash
cd src
python demo.py
```

## General training script structure
```python
# create trainer object
# It creates the model and the training and evaluation environments
trainer = Trainer( ... )  # parameters description on trainer.py docstring

# create list of training callbacks.
callbacks = [ ... ] # see 'src/callbacks/' or Stable Baselines3 docs

# train the model
trainer.train(
  tot_steps=<...>,  # number of overall training steps
  callbacks=callbacks,
  log_interval=<...>,  # number of steps between each log
  wandb=<...>,  # (bool) whether to use wandb logging
)
```

## Directories structure
- `NSPRs`: contains graphml files containing the definition of some Network Slice Placement Requests (NSPRs).
These can also be created on the fly during training, with no need to read files.

- `PSNs`: contains graphml files containing the definition of some Physical Substrate Networks (PSNs) architectures.

- `src`: contains the source code of the toolkit.
  
  - `callbacks`: contains some training callbacks.
  All callbacks in the library [Stable Baselines3](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) can be used as well.
  
  - `policies`: contains the implmentation of policy networks.
  It follows the nomenclature of [Stable Baselines3](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) policies, where the policy nets are composed of a features extractor followed by a MlpExtractor.
    - `features_extractors`: contains the implementation of features extractors modules.
    - `mlp_extractors`: contains the implementation of mlp extractors modules.
  
  - `spaces`: contains the implementation of custom [Gym](https://github.com/openai/gym) / [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) spaces.

  - `wrappers`: contains the implementation of custom environment wrappers.
  Wrappers from [Stable Baselines3](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) can also be used.

  - `network_simulator.py`: contains the implementation of the environment.

  - `trainer.py`: contains the implementation of the trainer object (see demo).

  - `demo.py`: contains a demo script.


## Contributing
Constributions are welcome! :rocket:

To contribute:
- If you want to **work on an open issue**, comment on that issue before opening a PR.
- If you want to implement a **new feature** or an **improvement**, write about it in the Discussions tab.

## Reference
```
Alex Pasquali, Vincenzo Lomonaco, Davide Bacciu and Federica Paganelli,
Deep Reinforcement Learning for Network Slice Placement and the DeepNetSlice Toolkit,
IEEE International Conference on Machine Learning for Communication and Networking, ICMLCN 2024, 5-8 May 2024, Stockholm, Sweden
```