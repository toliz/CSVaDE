# CSVaDE

Original implementation of **C**ommon **S**pace **Va**riational **D**eep **E**mbedding in PyTorch.

## Requirments

The code is tested with *Python 3.8.3*. A complete list of packages required is listed below:

| Package      | Version |
|--------------|---------|
| pytorch      | 1.5.1   |
| numpy        | 1.18.5  |
| scikit-learn | 0.23.1  |
| tensorboard  | 2.0.0   |

## Project Structure

```
├── README.md
│
├── configs
│   ├── embedding_size.json
│   ├── few-shot.json
│   ├── hidden_layers_1.json
│   ├── hidden_layers_2.json
│   ├── hidden_layers_3.json
│   ├── optimizer.json
│   ├── seen-unseen.json
│   └── zero-shot.json
│
├── datasets
│   ├── AWA1
│   ├── AWA2
│   ├── CUB
│   └── SUN
│
├── saved
│   └── csvade.pt
│
├── tensorboards
│   ├── experiments
│   └── models
│
├── main.py
├── models.py
├── train.py
├── data.py
└── utils.py
```

- `configs\` contains architecture configurations for various experiements (hyperparameter tuning)
- `datasets\` contains AWA1, AWA2, CUB and SUN datasets in pt (pytorch) file format
- `saved\` contains pretrained models
- `tensorboards\` contains tensorboard insights for pretained models
- *.py files contains the actual project code (start from main)

**Some of the above directories may not be present in the repo due to size restriction but will auto-generate when you run the code.*

**Link to data folder will be provided soon.*

## Run
You can either run in single mode using a single set of options:

`main.py --dataset AWA2 --num-shots 0 csvade`

or in batch mode for hyperparameter turning 

`main.py --dataset AWA2 --num-shots 0 configs/embedding_size.json`

for more options run:

`main.py -h`