import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from local_directories import vin_data_path

train_meta = pd.read_csv(vin_data_path / 'train_meta.csv').set_index('image_id')
train_df = pd.read_csv(vin_data_path / 'train.csv')
_, indices = np.unique(train_df.class_name, return_index=True)
unique_classes = train_df.class_name[indices].tolist()
unique_class_ids = train_df.class_id[indices].tolist()
class_id_from_class_name = dict(zip(unique_classes, unique_class_ids))

# Choose which class you want to analyze
chosen_class = st.sidebar.selectbox("Choose pathology", unique_classes, index=0)
chosen_class_id = class_id_from_class_name[chosen_class]

df = train_df[train_df.class_name == chosen_class]
image_ids = df.image_id.unique()

available_images = {
    str(file.with_suffix('').name)
    for file in (vin_data_path / 'train').glob('*.png')
}

assert available_images.intersection(image_ids)
if st.sidebar.button('Next'):
    random_image_id = np.random.choice(list(available_images.intersection(image_ids)))
else:
    random_image_id = np.random.choice(list(available_images.intersection(image_ids)))

image_path = vin_data_path / 'train' / (random_image_id+'.png')
bw_image = Image.open(image_path)
image = bw_image.convert("RGBA")
image_df = train_df[train_df.image_id == random_image_id]
radiologists = image_df.rad_id.unique()
pathology_df = image_df[image_df.class_name == chosen_class]
# Attention train_meta a stocké les dimensions numpy donc la première est les rows (y)
# et la seconde les colonnes (x)
original_height, original_width = train_meta.loc[random_image_id]
current_width, current_height = image.size
draw = ImageDraw.Draw(image)
for x_min, x_max, y_min, y_max, rad_id in \
        pathology_df[['x_min', 'x_max', 'y_min', 'y_max', 'rad_id']]\
                .itertuples(index=False, name=None):
    radiologist_color = 'red'
    new_x_min = int(x_min * current_width/original_width)
    new_x_max = int(x_max * current_width/original_width)
    new_y_min = int(y_min * current_height/original_height)
    new_y_max = int(y_max * current_height/original_height)
    draw.rectangle(((new_x_min, new_y_min), (new_x_max, new_y_max)), outline=radiologist_color)
    draw.text((new_x_min, new_y_min - 5), rad_id, fill=radiologist_color, anchor='ls',
              font=ImageFont.truetype('duran.otf', size=20))

st.image(image, use_column_width=True)

fig, ax = plt.subplots()
ax.bar(np.arange(256), bw_image.histogram(), width=1)
st.pyplot(fig)
st.sidebar.text(radiologists)
