import h5py
import numpy as np

# cross validation
valid_size = 500

label_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_last_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc_v2_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_arc_labels.npy"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc_labels.npy"
)

# (6885, 2)

labels = np.load(label_file)

print("Loaded Labels")

feature_h5_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_last.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc_v2.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_arc.h5"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc.h5"
)

test_feature_h5_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_bio_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_bio_test_truthful.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_bio_test_truthful.h5"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_bio_test_truthful.h5"
)

# embeddings: (13770, 31, 4096)

# best_acc = 0
# best_layer = None
# best_clf = None

val_results = []
all_results = []
best_scores = []

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
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    print("Start Logistic Regression")

    for layer in range(real_features.shape[0]):
        # clf = LogisticRegression(max_iter=100, fit_intercept=False)
        # clf = MLPClassifier(
        #     alpha=0.0,
        #     hidden_layer_sizes=(1,),
        #     activation="identity",
        #     random_state=42,
        #     max_iter=10,
        # )
        clf = MLPClassifier(
            alpha=0.0,
            hidden_layer_sizes=(32, 64),
            random_state=42,
            max_iter=5,
            batch_size=100,
        )
        # print("Fitting the model...")
        clf.fit(real_features[layer, valid_size:], real_labels[valid_size:])
        # print("Predicting...")
        pred = clf.predict(real_features[layer, :valid_size])
        # print("Calculate Accuracies...")
        acc = accuracy_score(real_labels[:valid_size], pred)
        print(f"layer {layer}: {acc}")

        # if acc > best_acc:
        #     best_acc = acc
        #     best_layer = layer
        #     best_clf = clf

        with h5py.File(test_feature_h5_file, "r") as f:
            test_features = f["embeddings"]
            print("Loaded Test Features")

            test_real_features = test_features
            # print(test_real_features.shape)
            features = test_real_features[layer]
            pred_score = clf.predict_proba(features)[:, 1]
            pred = (pred_score[::2] > pred_score[1::2]) * 1

            # labels is just all True
            labels = np.ones(pred.shape[0])
            acc = accuracy_score(labels, pred)
            print(f"Test Accuracy: {acc}")

        all_results.append(acc)
        best_scores.append((acc, pred_score))

    best_scores = sorted(best_scores, key=lambda x: x[0], reverse=True)

    for num_ensemble in range(2, 10):
        print(f"Ensemble {num_ensemble} models")
        ensemble_pred = np.zeros(len(pred) * 2)

        for i in range(num_ensemble):
            ensemble_pred += best_scores[i][1]

        ensemble_pred /= num_ensemble

        preds = []
        for i in range(0, len(ensemble_pred), 2):
            preds.append(1 if ensemble_pred[i] > ensemble_pred[i + 1] else 0)
        labels = np.ones(len(preds))

        acc = accuracy_score(labels, preds)
        print(f"Ensemble (={num_ensemble}) Acc:", np.mean(acc))

    print(all_results)
