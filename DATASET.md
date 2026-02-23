## Prepare data

Make a dataset folder
```bash
mkdir dataset
```

Download the original AVerImaTeC dataset using git
```bash
git clone https://huggingface.co/datasets/Rui4416/AVerImaTeC
```

download knowledge store using gdown
```bash
gdown --folder https://drive.google.com/drive/folders/1vjy7mjA4NTuLQfPh5-NZFpaxn8_H9rUs
```

Unzip all .zip files, and place them into the correct dataset folder. The structure is as follows:
```
┖ dataset
    ┖ AVerImaTeC (original averimatec dataset)
        ┖ train.json
        ┖ val.json
        ┖ images
            ┖ *.jpg
    ┖ AVerImaTeC_Shared_Task (original averimatec shared task knowledge store)
        ┖ Knowledge_Store
            ┖ train (knowledge store for training dataset)
                ┖ image_related/image_related_store_image_train
                    └ {i} (folder)
                        └ *.jpg
                └ text_related
                    └ image_related_store_text_train
                        └ {i}.json
                    └ text_related_store_text_train
                        └ {i}.json
            ┖ val (knowledge store for validation dataset, rename the folder name of converted_datastore (unzip from val.zip) -> val)
                ┖ image_related/image_related_store_image_val
                    └ {i} (folder)
                        └ *.jpg
                └ text_related
                    └ image_related_store_text_val
                        └ {i}.json
                    └ text_related_store_text_val
                        └ {i}.json
            ┖ test (knowledge store for validation dataset, rename the folder name of converted_datastore (unzip from test.zip) -> test)
                ┖ image_related/image_related_store_image_test
                    └ {i} (folder)
                        └ *.jpg
                └ text_related
                    └ image_related_store_text_test
                        └ {i}.json
                    └ text_related_store_text_test
                        └ {i}.json
        ┖ Vector_Store (vector for the embedding model)
```

## Knowledge store (Filled)

We provide a **filled `text_related` knowledge store** for the **validation** and **test** splits, along with a **Vector Store** for persistence.

Because generating embeddings for each knowledge store (for both **text** and **image**) can take a long time, we also release **precomputed embeddings** for:

- the **original** knowledge base  
- the **filled** knowledge base  

This release includes:
- the filled knowledge base files
- the Vector Store (for saving/loading)
- precomputed **text embeddings**
- precomputed **image embeddings**

You can download them from:  
https://huggingface.co/datasets/humane-lab/AVerImaTeC-Filled
