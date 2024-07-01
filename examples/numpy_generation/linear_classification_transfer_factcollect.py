import h5py
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score

# cross validation
valid_size = 500

label_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_factcollect_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_factcollect_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_factcollect_labels.npy"
)

# (6885, 2)

labels = np.load(label_file)

print("Loaded Labels")

feature_h5_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_factcollect.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_factcollect.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_factcollect.h5"
)

# embeddings: (13770, 31, 4096)

best_acc = 0
best_layer = None
best_clf = None

test_feature_h5_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_factcollect_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_covidfact_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_healthver_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_scifact_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_factcollect_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_covidfact_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_healthver_test.h5"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_scifact_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_factcollect_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_covidfact_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_scifact_test.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_healthver_test.h5"
)
test_label_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_factcollect_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_covidfact_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_healthver_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_scifact_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_factcollect_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_covidfact_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_healthver_test_labels.npy"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_scifact_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_factcollect_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_covidfact_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_scifact_test_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_healthver_test_labels.npy"
)

with open(test_label_file, "rb") as f:
    test_labels = np.load(f, allow_pickle=True)

with h5py.File(feature_h5_file, "r") as f:
    features = f["embeddings"]
    print("Loaded Features")

    print(features.shape, labels.shape)

    real_features = features
    real_labels = labels.reshape(-1)

    # real_features = features[:, ::2] - features[:, 1::2]
    # real_labels = labels.reshape(-1)[::2]

    # real_features: N x L x D
    # real_labels: N

    # experiment 1: train a linear classifier for each layer

    print(real_features[0, valid_size:].shape)
    print(real_features[0, :valid_size].shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    print("Start Logistic Regression")

    all_results_1 = []
    all_results_2 = []

    best_scores = []

    # for layer in range(real_features.shape[0]):
    for layer in range(1, real_features.shape[0]):
        clf = LogisticRegression(max_iter=100, fit_intercept=False)
        # clf = MLPClassifier(
        #     alpha=0.0,
        #     hidden_layer_sizes=(1,),
        #     activation="identity",
        #     random_state=42,
        #     max_iter=10,
        # )
        # clf = MLPClassifier(
        #     alpha=0.0,
        #     hidden_layer_sizes=(32, 64),
        #     random_state=42,
        #     max_iter=5,
        #     batch_size=100,
        # )

        clf.fit(real_features[layer, valid_size:], real_labels[valid_size:])
        pred = clf.predict(real_features[layer, :valid_size])

        # # coef_ndarray of shape (1, n_features) or (n_classes, n_features)
        # # replace coef_ndarray with the mean of real_features
        # # print(clf.coef_.shape) # (1, 4096)
        # clf.coef_ = np.mean(real_features[layer, valid_size:], axis=0, keepdims=True)

        # print("Calculate Accuracies...")
        acc = accuracy_score(real_labels[:valid_size], pred)
        print(f"layer {layer}: {acc}")

        # all_results_1.append(acc)

        best_layer = layer
        best_clf = clf

        with h5py.File(test_feature_h5_file, "r") as f:
            test_features = f["embeddings"]
            test_real_features = test_features
            features = test_real_features[best_layer]

            pred = best_clf.predict_proba(features)[:, 1]

            if "factcollect" in test_feature_h5_file:
                micro_f1 = f1_score(test_labels, pred > 0.5, average="micro")
            else:
                micro_f1 = f1_score(test_labels, pred > 0.5)
            bacc = balanced_accuracy_score(y_true=test_labels, y_pred=pred > 0.5)

        print("Test F1:", micro_f1)
        print("Test BACC:", bacc)

        all_results_1.append(micro_f1)
        all_results_2.append(bacc)

        best_scores.append((bacc + micro_f1, pred))

    best_scores = sorted(best_scores, key=lambda x: x[0], reverse=True)

    for num_ensemble in range(2, 10):
        print(f"Ensemble {num_ensemble} models")
        ensemble_pred = np.zeros(len(pred))

        for i in range(num_ensemble):
            ensemble_pred += best_scores[i][1]

        ensemble_pred /= num_ensemble

        if "factcollect" in test_feature_h5_file:
            f1 = f1_score(test_labels, ensemble_pred > 0.5, average="micro")
        else:
            f1 = f1_score(test_labels, ensemble_pred > 0.5)
        bacc = balanced_accuracy_score(y_true=test_labels, y_pred=ensemble_pred > 0.5)

        print(f"Ensemble (={num_ensemble}) BACC:", bacc)
        print(f"Ensemble (={num_ensemble}) F1:", f1)

    print(all_results_1)
    print(all_results_2)
