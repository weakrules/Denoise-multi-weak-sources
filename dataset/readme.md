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

### `*_all_LF.csv`
It consists of original text, weak labels, and ground turth labels.

For the full preprocessed dataset, it is available to download at https://drive.google.com/drive/folders/1MJe1BJYNPudfmpFxCeHwYqXMx53Kv4h_?usp=sharing. 