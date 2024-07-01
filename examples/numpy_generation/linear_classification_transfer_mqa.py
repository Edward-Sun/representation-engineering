import h5py
import numpy as np

# cross validation
valid_size = 500

label_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_last_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_entity_plus_last_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts_last_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts_v2_last_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_facts_v2_last_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc_v2_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_arc_labels.npy"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc_labels.npy"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc_v2_labels.npy"
)

# (6885, 2)

labels = np.load(label_file)

print("Loaded Labels")

feature_h5_file = (
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_entity_plus_last.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_entity_plus_last.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts_last.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_facts_v2_last.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_facts_v2_last.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_arc_v2.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_arc.h5"
    "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc.h5"
    # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_arc_v2.h5"
)

# embeddings: (13770, 31, 4096)

best_acc = 0
best_layer = None
best_clf = None

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
    from sklearn.metrics import accuracy_score

    print("Start Logistic Regression")

    all_results_1 = []
    all_results_2 = []

    best_scores = []

    # for layer in range(real_features.shape[0]):
    for layer in range(10, real_features.shape[0]):
        clf = LogisticRegression(max_iter=100, fit_intercept=False)
        # print("Fitting the model...")

        # clf = MLPClassifier(
        #     alpha=0.0,
        #     hidden_layer_sizes=(32, 64),
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

        # print("Predicting...")
        pred = clf.predict(real_features[layer, :valid_size])

        # # coef_ndarray of shape (1, n_features) or (n_classes, n_features)
        # # replace coef_ndarray with the mean of real_features
        # # print(clf.coef_.shape) # (1, 4096)
        # clf.coef_ = np.mean(real_features[layer, valid_size:], axis=0, keepdims=True)

        # print("Calculate Accuracies...")
        acc = accuracy_score(real_labels[:valid_size], pred)
        print(f"layer {layer}: {acc}")

        all_results_1.append(acc)

        # if acc > best_acc:
        #     best_acc = acc
        #     best_layer = layer
        #     best_clf = clf

        best_layer = layer
        best_clf = clf

        test_feature_h5_file = (
            # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_medqa_v3.h5"
            # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-7b_medqa_v4.h5"
            # "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-13b_medqa_v3.h5"
            "/home/zhengbaj/tir5/zhiqing/data/dataset_embed/llama-2-70b_medqa_v3.h5"
        )

        with h5py.File(test_feature_h5_file, "r") as f:
            test_features = f["embeddings"]
            test_real_features = test_features
            features = test_real_features[best_layer]

            pred = best_clf.predict_proba(features)[:, 1]

        truthfulqa_label_file = (
            "/home/zhengbaj/tir5/zhiqing/data/dataset_medqa/medqa_test.json"
        )

        import json

        with open(truthfulqa_label_file, "r") as f:
            truthfulqa_labels = json.load(f)

        acc = []
        label_cnt = 0

        for example in truthfulqa_labels:
            labels = example["labels"]
            pred_labels = pred[label_cnt : label_cnt + len(labels)]

            pred_choice = int(np.argmax(pred_labels))

            # if index max of pred_label is 1 in labels, then it is correct
            acc.append(1 if labels[pred_choice] == 1 else 0)
            label_cnt += len(labels)

        print("TruthfulQA Acc:", np.mean(acc))

        all_results_2.append(np.mean(acc))

        best_scores.append((np.mean(acc), pred))

    best_scores = sorted(best_scores, key=lambda x: x[0], reverse=True)

    for num_ensemble in range(2, 10):
        print(f"Ensemble {num_ensemble} models")
        ensemble_pred = np.zeros(len(pred))

        for i in range(num_ensemble):
            ensemble_pred += best_scores[i][1]

        ensemble_pred /= num_ensemble

        acc = []
        label_cnt = 0

        for example in truthfulqa_labels:
            labels = example["labels"]
            pred_labels = ensemble_pred[label_cnt : label_cnt + len(labels)]

            pred_choice = int(np.argmax(pred_labels))

            # if index max of pred_label is 1 in labels, then it is correct
            acc.append(1 if labels[pred_choice] == 1 else 0)
            label_cnt += len(labels)

        print(f"Ensemble (={num_ensemble}) Acc:", np.mean(acc))

    print(all_results_1)
    print(all_results_2)
