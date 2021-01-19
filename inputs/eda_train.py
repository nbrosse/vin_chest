import pandas as pd

from local_directories import vin_data_path

train = pd.read_csv(str(vin_data_path / "train.csv"))
images_classes = train[["image_id", "class_name"]]
images_classes = images_classes.value_counts().reset_index()