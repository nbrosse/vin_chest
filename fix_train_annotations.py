import numpy as np
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

image_path = vin_data_path / 'train' / (random_image_id + '.png')
bw_image = Image.open(image_path)
image_df = train_df[train_df.image_id == random_image_id]
radiologists = image_df.rad_id.unique()
pathology_df = image_df[image_df.class_name == chosen_class]
# Attention train_meta a stocké les dimensions numpy donc la première est les rows (y)
# et la seconde les colonnes (x)
original_height, original_width = train_meta.loc[random_image_id]
current_width, current_height = bw_image.size


def plot_bboxes(image):
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    for x_min, x_max, y_min, y_max, rad_id in \
            pathology_df[['x_min', 'x_max', 'y_min', 'y_max', 'rad_id']] \
                    .itertuples(index=False, name=None):
        radiologist_color = 'red'
        new_x_min = int(x_min * current_width / original_width)
        new_x_max = int(x_max * current_width / original_width)
        new_y_min = int(y_min * current_height / original_height)
        new_y_max = int(y_max * current_height / original_height)
        draw.rectangle(((new_x_min, new_y_min), (new_x_max, new_y_max)), outline=radiologist_color)
        try:
            font = ImageFont.truetype('duran.otf', size=20)
        except:
            font = ImageFont.load_default()
        draw.text((new_x_min, new_y_min - 5), rad_id, fill=radiologist_color, anchor='ls',
                  font=font)
        st.sidebar.text(
            f"{rad_id} ({(x_min + x_max) / 2 / original_width:.1%} / {(y_min + y_max) / 2 / original_height:.1%})")
    return image


def get_better_contrast(bw_image):
    for relative_size in [0.5]:
        x_min, x_max = current_width * (0.5 - relative_size / 2), current_width * (0.5 + relative_size / 2)
        y_min, y_max = current_height * (0.5 - relative_size / 2), current_height * (0.5 + relative_size / 2)
    croped_image = bw_image.crop((x_min, y_min, x_max, y_max))
    print(croped_image.size)
    min_light = min(croped_image.getdata())
    max_light = max(croped_image.getdata())
    img_pixels = np.array(bw_image)
    better_contrast_pixels = (img_pixels.astype(np.float) - min_light) / (max_light - min_light) * 256
    better_contrast_pixels[better_contrast_pixels <= 0] = 0
    better_contrast_pixels[better_contrast_pixels >= 255] = 255
    better_contrast_pixels = better_contrast_pixels.astype(np.uint8)
    return Image.fromarray(better_contrast_pixels)


def write_rad_decisions_in_the_sidebar():
    for rad_id in image_df[image_df.class_id == 14].rad_id.values:
        st.sidebar.text(f"{rad_id}")
        if st.sidebar.button('Ignore him'):
            st.sidebar.text("Well done !")


better_contrast_img = get_better_contrast(bw_image)
better_contrast_img = plot_bboxes(better_contrast_img)
write_rad_decisions_in_the_sidebar()
st.image(better_contrast_img, use_column_width=True)
st.sidebar.text(radiologists)
