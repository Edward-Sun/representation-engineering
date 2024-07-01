import h5py
import numpy as np

# cross validation
valid_size = 500

label_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_labels.npy"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_last_labels.npy"
)

# (6885, 2)

labels = np.load(label_file)

print("Loaded Labels")

feature_h5_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus.h5"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_last.h5"
)

# embeddings: (13770, 31, 4096)

with h5py.File(feature_h5_file, "r") as f:
    features = f["embeddings"]
    print("Loaded Features")

    print(features.shape, labels.shape)

    real_features = features
    real_labels = labels.reshape(-1)

    # real_features: N x L x D
    # real_labels: N

    # experiment 1: train a linear classifier for each layer

    print(real_features[0, valid_size:].shape)
    print(real_features[0, :valid_size].shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("Start Logistic Regression")

    for layer in range(real_features.shape[0]):
        clf = LogisticRegression(max_iter=1000)
        # print("Fitting the model...")
        clf.fit(real_features[layer, valid_size:], real_labels[valid_size:])
        # print("Predicting...")
        pred = clf.predict(real_features[layer, :valid_size])
        # print("Calculate Accuracies...")
        acc = accuracy_score(real_labels[:valid_size], pred)
        print(f"layer {layer}: {acc}")

    # # check acc on combined features of layer 12, 14, 16

    # combined_features = np.concatenate(
    #     [
    #         real_features[12, :],
    #         real_features[14, :],
    #         real_features[16, :],
    #     ],
    #     axis=1,
    # )

    # valid_combined_features = combined_features[:valid_size]
    # train_combined_features = combined_features[valid_size:]

    # clf = LogisticRegression(max_iter=2000)
    # clf.fit(train_combined_features, real_labels[valid_size:])
    # pred = clf.predict(valid_combined_features)
    # acc = accuracy_score(real_labels[:valid_size], pred)
    # print(f"combined features: {acc}")
