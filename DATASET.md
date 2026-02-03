## Prepare data

make dataset folder
```bash
mkdir dataset
```

download original averimatec dataset using git
```bash
git clone https://huggingface.co/datasets/Rui4416/AVerImaTeC
```

download knowledge store using gdown
```bash
gdown --folder https://drive.google.com/drive/folders/1vjy7mjA4NTuLQfPh5-NZFpaxn8_H9rUs
```

Unzip all .zip file, and place them into correct dataset folder. The structure are follows:
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
```