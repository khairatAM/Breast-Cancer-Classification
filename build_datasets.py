from cancernet import config
from imutils import paths
import random, shutil, os

original_paths = list(paths.list_images(config.INPUT_DATASET))
random.seed(42)
random.shuffle(original_paths)

index = int(len(original_paths)*config.TRAIN_SPLIT)
train_paths = original_paths[:index]
test_paths = original_paths[index:]

index = int(len(train_paths)*config.VAL_SPLIT)
val_paths = train_paths[index:]
train_paths = train_paths[:index]

datasets = [("training", train_paths, config.TRAIN_PATH),
            ("validation", val_paths, config.VAL_PATH),
            ("testing", test_paths, config.TEST_PATH)]

for (set_type, original_paths, base_path) in datasets:
    print(f'Building {set_type} set')

    if not os.path.exists(base_path):
        print(f'Building directory {base_path}')
        os.makedirs(base_path)

    for path in original_paths:
        file=path.split(os.path.sep)[-1] #get the name of each file
        label=file[-5:-4] #get the class/label of each file (embedded in the filename)
        label_path = os.path.sep.join([base_path, label])

        if not os.path.exists(label_path):
            print(f'Building directory {label_path}')
            os.makedirs(label_path)

        new_path = os.path.sep.join([label_path, file])
        shutil.copy2(path,new_path)
        
