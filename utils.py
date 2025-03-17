import numpy as np
import medmnist
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import time


def load_data(dataset_folder, data_flag, size_flag):
    npz_file = np.load(f'{dataset_folder}/{data_flag}_{size_flag}.npz', mmap_mode='r')
    train_images = npz_file['train_images']
    train_labels =  npz_file['train_labels']
    valid_images = npz_file['val_images']
    valid_labels =  npz_file['val_labels']
    test_images = npz_file['test_images']
    test_labels =  npz_file['test_labels']

    return [train_images, train_labels, valid_images, valid_labels, test_images, test_labels]


# def create_loaders(data_flag, image_size, data_transform, batch_size=128, download=False):
#     dataset_folder = f'./{data_flag}'
#     info = medmnist.INFO[data_flag]
#     DataClass = getattr(medmnist, info['python_class'])

#     train_dataset = DataClass(root=dataset_folder, split='train', size=image_size, transform=data_transform, download=download)
#     valid_dataset = DataClass(root=dataset_folder, split='val', size=image_size, transform=data_transform, download=download)
#     test_dataset = DataClass(root=dataset_folder, split='test', size=image_size, transform=data_transform, download=download)

#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#     valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, valid_loader, test_loader


def cal_metrics(y_true, y_score, threshold=0.5):
    y_pre = y_score > threshold
    acc, f1, auc = 0, 0, 0
    for label in range(y_true.shape[1]):
        label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
        label_f1 = f1_score(y_true[:, label], y_pre[:, label])
        label_auc = roc_auc_score(y_true[:, label], y_score[:, label])
        acc += label_acc
        f1 += label_f1
        auc += label_auc

    acc = acc / y_true.shape[1]
    f1 = f1 / y_true.shape[1]
    auc = auc / y_true.shape[1]

    return [acc, f1, auc]


def train(model, X_train, Y_train, n_epochs, batch_size, model_path=None):
    is_improved = True
    not_improved_count = 0
    best_auc = 0
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        model.fit(X_train, Y_train,
                  epochs=1,
                  batch_size=batch_size
                  )
        train_metrics = evaluate(model, X_train, Y_train)
        if train_metrics[2] > best_auc:
            best_auc = train_metrics[2]
            if model_path != None:
                model.save_weights(model_path)
        else:
            not_improved_count +=1
            if not_improved_count == 5:
                break

        print(f'train_acc: {train_metrics[0]}; train_f1: {train_metrics[1]}; train_auc: {train_metrics[2]}')
    print('Model saved to:', model_path)


def evaluate(model, X_test, Y_test):
    y_score = model.predict(X_test)
    test_metrics = cal_metrics(Y_test, y_score)
    
    return test_metrics


def cal_throughput(model, X_test):
    t1 = time.time()
    model.predict(X_test)
    t2 = time.time()
    throughput = X_test.shape[0] / (t2 - t1)

    return throughput