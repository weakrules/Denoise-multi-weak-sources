# Multi-Source-Weak-Supervision

**Learning from Multi-SourceWeak Supervision for Deep Text Classification**

## Code

### Environment
Python 3.6

The code can be run on either CPU or GPU environment.

### Training the model and make predictions
To run the model, first unzip the dataset file, and then using either way is fine:

(Note: due to the github space limitation, we only include the three dataset. 
The entire dataset can be downloaded using: https://drive.google.com/drive/folders/1MJe1BJYNPudfmpFxCeHwYqXMx53Kv4h_?usp=sharing)

(1) python main_conditional_attn.py --ds {$dataset}

(For example: python main_conditional_attn.py --ds imdb)

```css
usage: main_conditional_attn.py [-h] [--pt_file PT_FILE] --ds
                                {youtube,imdb,yelp,agnews,spouse} [--no_cuda]
                                [--fast_mode] [--seed SEED] [--epoch EPOCH]
                                [--lr LR] [--weight_decay WEIGHT_DECAY]
                                [--hidden HIDDEN] [--c2 C2] [--c3 C3]
                                [--c4 C4] [--k K] [--x0 X0]
                                [--unlabeled_ratio UNLABELED_RATIO]
                                [--log_prefix LOG_PREFIX] [--ft_log FT_LOG]
                                [--n_high_cov N_HIGH_COV]
```

(2) sh run_conditional.sh

The trained model will be stored at the *model* folder. 

The running details output can be found at *log_files* folder.

The test accuracy can be found at *ft_logs* folder.

## Dataset
Dataset:
- agnews
- imdb
- spouse
- yelp
- youtube

The required data are stored as *.pt file, and each record includes the following information:
   - the original document text ('text')
   - the extracted pre-trained Transformer feature ('bert_feature'')
   - the ground truth label ('label')
   - the annotated noisy labels ('lf')
   - the simple majority voting label of annotated noisy labels ('major_label')

We use a dictionary to store the training, validation, and test data.
The division are maintained the same for all the baselines as well.

### `*_organized_nb.pt`

```python
data_dict = {
    'labeled': {
        'text':
        'label': 
        'major_label': 
        'lf': 
        'bert_feature': 
    },
    'unlabeled': {
        'text': 
        'label': 
        'major_label': 
        'lf': 
        'bert_feature': 
    },
    'test': {
        'text': 
        'label': 
        'major_label':
        'lf': 
        'bert_feature':
    },
    'validation': {
        'text': 
        'label': 
        'major_label': 
        'lf': 
        'bert_feature':
    }
}
```

## Labeling sources with rules and annotated labels

We provide **Labeling Functions** and **Labeling Results** of each dataset in the *rules-noisy-labels* folder.
The specific description is included in the inside README.

 
