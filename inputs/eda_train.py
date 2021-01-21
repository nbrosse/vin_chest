import pandas as pd

from local_directories import vin_data_path, xp_path


def load_train() -> pd.DataFrame:
    train = pd.read_csv(str(vin_data_path / "train.csv"))
    train = train.sort_values(by="image_id")
    dict_image_id_to_image_nb = {image_id: i for i, image_id in enumerate(train.image_id.unique())}
    train.loc[:, "image_nb"] = train.image_id.map(dict_image_id_to_image_nb)
    train = train[["image_nb", "class_name", "class_id", "rad_id", "x_min", "y_min", "x_max", "y_max"]]
    return train


def statistics_by_images_and_classes(train: pd.DataFrame) -> None:
    images_classes = train[["image_nb", "class_name"]]
    images_classes = images_classes.value_counts().reset_index().rename(columns={0: "number"}).sort_values("image_nb")

    summary_by_class = {
        class_name: {
            "image_nb": list(),
            "ratio": list(),
            "nb_by_image": list(),
            "total_nb_by_image": list(),
        } for class_name in images_classes.class_name.unique()
    }

    for image_nb, df_image_nb in images_classes.groupby("image_nb"):
        total_count = df_image_nb["number"].sum()
        for _, row in df_image_nb.iterrows():
            summary_by_class[row["class_name"]]["image_nb"].append(row["image_nb"])
            summary_by_class[row["class_name"]]["ratio"].append(row["number"] / total_count)
            summary_by_class[row["class_name"]]["nb_by_image"].append(row["number"])
            summary_by_class[row["class_name"]]["total_nb_by_image"].append(total_count)

    summary_by_class = {
        class_name: pd.DataFrame(d).sort_values("image_nb") for class_name, d in summary_by_class.items()
    }

    list_dfs_described_by_class = list()

    for class_name, df_class in summary_by_class.items():
        df_described_by_class = df_class[["ratio", "nb_by_image", "total_nb_by_image"]].describe()
        df_described_by_class.loc[:, "class_name"] = class_name
        list_dfs_described_by_class.append(df_described_by_class)

    df_described_by_class = pd.concat(list_dfs_described_by_class, axis=0)
    del list_dfs_described_by_class
    df_described_by_class = df_described_by_class[["class_name", "ratio", "nb_by_image", "total_nb_by_image"]]
    df_described_by_class.to_csv(str(xp_path / "df_described_by_class.csv"), index=True)

train = load_train()
train_abnormalities = train[train.class_name != "No finding"]
train_abnormalities = train_abnormalities.sort_values(by=["image_nb", "class_name"])
train_abnormalities.to_csv(str(xp_path / "train_abnormalities.csv"), index=False)

# train_abnormalities["image_nb"].value_counts().describe()
# train_abnormalities.groupby("class_name")["image_nb"].value_counts().rename("number").reset_index().drop(columns=["image_nb"]).groupby("class_name").describe()

# def

# train_abnormalities.groupby(by=["image_nb", "class_name"]).apply()
