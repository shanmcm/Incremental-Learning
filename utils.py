import random
import statistics
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from data import CustomCifar100
from resnet32 import CifarResNet
# from models import *

def get_training_test_batches(mode):
    '''
    Handles the initialization of the dataset

    Returns:
        train & test batches, lists containing (item, target) pairs
    '''
    data_dir = '.'
    training_data = CustomCifar100(data_dir, True, transform=transforms.ToTensor())
    test_data = CustomCifar100(data_dir, False, transform=transforms.ToTensor())

    print(f'Train Dataset: {len(training_data)}')
    print(f'Test Dataset: {len(test_data)}')


    train_batches, test_batches = [], []
    for i in range(10):
        train_batch_indexes = training_data.indexes_per_batch[i]

        test_batch_indexes = []
        if mode == "default":
            for j in range(i+1):
                test_indexes = test_data.indexes_per_batch[j]
                test_batch_indexes += test_indexes
        elif mode in ('cwwr', 'cwwnr'):
            if i>=5:
                break
            for j in range(i+1):
                test_indexes = test_data.indexes_per_batch[j]
                test_batch_indexes += test_indexes
        elif mode in ('ownr', 'owr'):
            if i>=5:
                break
            else: 
                for j in range(5, 5+i+1):
                    test_indexes = test_data.indexes_per_batch[j]
                    test_batch_indexes += test_indexes

        train_batch = Subset(training_data,train_batch_indexes)
        test_batch = Subset(test_data,test_batch_indexes)

        train_batches.append(train_batch)
        test_batches.append(test_batch)

    return train_batches, test_batches

def initialize_and_get_args(args: dict = None, model_type = "resnet32"):
    '''
    Calls set_seeds() and return default argument dict of static variables

    Returns:
        args: dict with static variables
    '''
    set_seeds()
    if args is None:
        args: dict = dict()
        args["lr"] = 2
        args["wd"] = 1e-4
        args["mom"] = 0
        args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        args["batch_size"] = 128
        args["memory"] = 2000
        if model_type in ("resnet20", "resnet32", "resnet44", "resnet56", "resnet110"):
            args["feature_size"] = 64
        else:
            print("Model {mode_str} not supported. Exiting...")
            exit()
            

    return args


def set_seeds(seed: int = 3):
    '''
    Sets the seeds for numpy, random, torch_manual and torch.cuda

    Arguments:
        seed: int
    '''
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def get_model(model_str: str):
    if model_str == 'resnet32':
        return CifarResNet()
    elif model_str == 'resnet20':
        return CifarResNet(depth=20)
    elif model_str == 'resnet44':
        return CifarResNet(depth=44)
    elif model_str == 'resnet56':
        return CifarResNet(depth=56)
    elif model_str == 'resnet110':
        return CifarResNet(depth=110)
    else:
        print("Model {mode_str} not supported. Exiting...")
        exit()

def confusionMatrix(labels, predictions, step):
    '''
    Print the confusion matrix given labels and predictions

    Arguments:
        labels: ground truth values
        predictions: predicted values
        step: used to compute the # of ticks neede
    '''
    ticks = np.arange(10, 110, 10)
    plt.figure(figsize=(8, 8))
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(np.log(cm+1), cmap='jet', cbar=False)
    plt.xticks(ticks[:step+1], labels=ticks[:step+1], rotation='horizontal')
    plt.yticks(ticks[:step+1], labels=ticks[:step+1], rotation='horizontal')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()


def get_oneHot(outputs, y):
    oh_y = torch.zeros(outputs.shape[0], outputs.shape[1])
    oh_y[range(oh_y.shape[0]), y.long()] = 1
    return oh_y


def get_oneHot_labels(target, n_classes: int):
    '''
    Returns the target values in the one hot encoding with n_classes possible values

    Arguments:
        labels: ground truth values
        predictions: predicted values
        step: used to compute the # of ticks neede
    Returns:
        one_hot: encoded target
    '''
    one_hot = torch.zeros(target.shape[0], n_classes)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


def fc_classifier(test_batch, model, n_classes, args, mode="default"):
    model.eval()

    '''
    mode:
    - cwwr: closed world with rejection
    - owr: open world with rejection

    '''
    correct, rejected = 0, 0
    kept_labels, kept_predictions, all_labels = [], [], []
    rej_thresh = 0.5
    accepted, rejected, total = 0, 0, 0
    unknown_label=-1
    with torch.no_grad():
        loader = DataLoader(test_batch, batch_size=args["batch_size"], shuffle=False, num_workers=2, drop_last=False)
        for images, labels in loader:
            total+=len(labels)
            images = images.to(args["device"])
            labels = labels.to(args["device"])
           
            # Il problema era qui, mancava vincolo della dimensione
            outputs = torch.softmax(model(images), dim=1, dtype = images.dtype)
            # Get predictions
            probabilities, predictions = torch.max(outputs.data, 1)
            # Update Corrects
           
            for index, (probability, prediction, label) in enumerate(zip(probabilities, predictions, labels)):
                if mode == 'default':
                    kept_predictions.append(np.array(prediction.cpu()))
                    kept_labels.append(np.array(label.cpu()))
                #cwwr=closed world 
                else:
                    if probability > rej_thresh:
                        accepted+=1
                        if mode=='cwwr':
                            if prediction==label:
                              correct+=1
                    else:
                        rejected+=1
                        if mode=='owr':
                            correct+=1

    # Compute Accuracy
    if mode == 'default':
        accuracy=accuracy_score(kept_labels, kept_predictions)
        print(f'   # FC Layer Accuracy: {accuracy:.2f}')
    elif mode=='owr':
        acc=accepted/total
        rej=rejected/total
        accuracy=correct/total
        print(f'accepted: {accepted}')
        print(f'rejected (and correct): {rejected}')
        print(f'total: {total}')
        print(f'Accuracy as correct/total: {accuracy}')
    else: #cwwr
        acc=accepted/total
        rej=rejected/total
        accuracy=correct/total
        print(f'accepted: {accepted}')
        print(f'correctly accepted: {correct}')
        print(f'rejected: {rejected}')
        print(f'total: {total}')
        print(f'Accuracy as correct/total: {accuracy}')

    return accuracy, kept_predictions, kept_labels

def get_max_distance_per_class(features, means):
    classes = list(means.keys())
    dist_by_class = dict((k, []) for k in classes)
    max_dist_by_class = dict.fromkeys(classes, 0)
    
    for label, features_by_class in features.items():
        for feature in features_by_class:
            for class_label, mean in means.items():
                if class_label == label:
                    item_dist = torch.dist(mean, feature)
                    dist_by_class[class_label].append(item_dist)

              
    percentile=25
    for class_label in dist_by_class.keys():
        p = int(len(dist_by_class[class_label]) * percentile / 100)
        max_dist_by_class[class_label]=sorted(dist_by_class[class_label])[p]        
    return max_dist_by_class


def get_min_distance_per_class(output, means, label):
  
    classes=list(means.keys())
    for class_label, mean in means.items():
        if class_label==label: #label is the one we gave
            dist=torch.dist(output, mean)
    return dist

def nme_classifier(test_batch, train_batch, exemplars, model, n_classes, args, mode):
    '''
    Nearest Mean of Exemplars Classifier

    Parameters:
        train_batch: batch used for training
        test_batch: batch used for testing
        model: the resnet used
        args: dict containing the static variables

    mode:
    - deafult: iCaRL
    - ownr: Open world with non naive rejection strategy based on distances
    - cwwnr: Closed world with non naive rejection strategy based on distances
    '''
    model.eval()

    means = dict.fromkeys(np.arange(n_classes))
    features: dict = dict((k, []) for k in range(n_classes))
    class_map: dict = dict((k, []) for k in range(n_classes-10, n_classes))
    rejected = dict.fromkeys(np.arange(n_classes), 0)

    # Set value part to empy list
    for item in train_batch:
        for key in class_map.keys():
            # if the item label is equal to key (label), then add the item to the list associated to that label
            if item[1] == key:
                class_map[key].append(item)

    # Compute means
    for key in range(n_classes):
        # If I am in the last ten classes use data
        if key in range(n_classes-10, n_classes):
            items = class_map[key]
        # otherwise refer to exemplars
        else:
            items = exemplars[key]

        # loader with defined items
        loader = DataLoader(items, batch_size=args["batch_size"], shuffle=True, num_workers=2, drop_last=False)

        feat_sum = torch.zeros((1, args["feature_size"]), device=args["device"])  # 64 for resnet structure
        for images, labels in loader:
            with torch.no_grad():
                images = images.to(args["device"])
                # Perform some data augmentation
                flipped_images = torch.flip(images, [3])
                # Concatenate images and flipped version
                images = torch.cat((images, flipped_images))
                outputs = model(images, features=True)  # outputs features instead of labels (1x64 instead of 32x32)

                for output in outputs:
                    feat_sum += output
                    features[key].append(output)

        # feat_sum is 1x64 tensor obtained by summing all outputs
        mean = feat_sum/(2*len(items))  # I divide by 2*len because i concatenated images with its flipped version => dimension doubles
        # Add mean to means dictionary (and normalize, good practise to do so)
        means[key]=mean/mean.norm()

    loader = DataLoader(test_batch, batch_size=args["batch_size"], shuffle=True, num_workers=2, drop_last=False)
    predictions, label_list = [], []
    if mode in ("ownr", "cwwnr"):
        rej_thresh = get_max_distance_per_class(features, means)
    
    correct, accepted, total= 0, 0, 0
    for images, labels in loader:
        total+=len(labels)
        images = images.to(args["device"])
        to_pop = list()
        with torch.no_grad():
            outputs = model(images, features=True)
            for idx, output in enumerate(outputs):
                pred = None
                min_dist = 10**4
                # iterate over classes (keys) of mean
                for key in means.keys():
                    # calculate the distance between the ouput and average feature vector (evaluated in key)
                    dist = torch.dist(means[key], output)

                    if dist < min_dist:
                        min_dist = dist
                        pred = key  # update pred with class key that (so far) minimises distance
                # print(min_dist)
                if mode == "default":
                    predictions.append(pred)
                else:
                    maximum=rej_thresh[pred] #calculate distance between centroid and its elements and select maximum
                    dist_centroid_output=get_min_distance_per_class(output, means, pred) #calculate distance between centroid and pred
                   
                    if dist_centroid_output > maximum:
                        rejected[pred] += 1
                        to_pop.append(idx)
                        if mode == "ownr":
                            correct+=1
                    else:
                        accepted+=1
                        predictions.append(pred)
                        if mode == "cwwnr":
                            correct += 1

            if mode in ("ownr", "cwwnr"): 
                labels = list(labels)
                for index in sorted(to_pop, reverse=True):  # needs to be reversed otherwiese deleting items changes the subsequent ones
                    del labels[index]

        for label in labels:
            label_list.append(label)
        #label_list += list(labels)

    if mode == "default":
        accuracy = accuracy_score(label_list, predictions)
    elif mode == "ownr":
        accuracy=correct/total
        print(f'accepted: {accepted}')
        print(f'rejected (and correct): {correct}')
        print(f'total: {total}')
        print(f'Accuracy as correct/total: {accuracy}')
    elif mode == "cwwnr":
        accuracy=correct/total
        rejected=sum({value for (key, value) in rejected.items()})
        print(f'accepted (and correct): {accepted}')
        print(f'rejected: {rejected}')
        print(f'total: {total}')
        print(f'Accuracy as correct/total: {accuracy}')

    return accuracy, predictions, label_list


def svm_classifier(exemplars,test, model, args):

    model.eval()
    X_train, Y_train = [], []
    
    new_exemplars=[]
    for key in exemplars:
        for item in exemplars[key]:
            new_exemplars.append([item[0], item[1]])
  
    # Define train data
    loader = DataLoader(new_exemplars, batch_size=args["batch_size"], shuffle=False, num_workers=2, drop_last=False)

    for images, labels in loader:
        with torch.no_grad():
            images = images.to(args["device"])
            # Perform some data augmentation
            flipped_images = torch.flip(images, [3])
            # Concatenate images and flipped version
            images = torch.cat((images, flipped_images))
            outputs = model(images, features=True)  # outputs features instead of labels (1x64 instead of 32x32)

            for output, label in zip(outputs, labels):
                X_train.append(np.array(output.cpu()))
                Y_train.append(np.array(label))

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, train_size=0.8, random_state=0)
    score = 'accuracy'
    svm = SVC()

    grid = {'C': [0.1], 'kernel': ['rbf']}  # Tried with 'C':[0.001, 0.01, 0.1, 1], C=1 is the best

    best = best_version(x_train, y_train, x_val, y_val, svm, score, grid, -1)
    svm = SVC(**best.best_params_)

    svm.fit(X_train, Y_train)

    # Work on test data and predict
    loader = DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=2, drop_last=False)

    predictions, label_list = [], []
    for images, labels in loader:
        images = images.to(args["device"])
        label_list += labels
        with torch.no_grad():
            outputs = model(images, features=True)
            for output in outputs:
                prediction = svm.predict([np.array(output.cpu())])
                predictions.append(prediction)

    accuracy = accuracy_score(label_list, predictions)
    return accuracy, predictions, label_list


def knn_classifier(exemplars, test, model, args):
 
    model.eval()
    X_train, Y_train = [], []

    new_exemplars=[]
    for key in exemplars:
        for item in exemplars[key]:
            new_exemplars.append([item[0], item[1]])

    # Define train data
    loader = DataLoader(new_exemplars, batch_size=args["batch_size"], shuffle=True, num_workers=2, drop_last=False)

    for images, labels in loader:
        with torch.no_grad():
            images = images.to(args["device"])
            # Perform some data augmentation
            flipped_images = torch.flip(images, [3])
            # Concatenate images and flipped version
            images = torch.cat((images, flipped_images))
            outputs = model(images, features=True)  # outputs features instead of labels (1x64 instead of 32x32)

            for output, label in zip(outputs, labels):
                X_train.append(np.array(output.cpu()))
                Y_train.append(np.array(label))

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, train_size=0.8, random_state=0)
    score = 'accuracy'
    knn = KNeighborsClassifier()

    grid = {'n_neighbors': [10]}  # Tried with [3,4,5,6,7,8,9,10], 10 is the best

    best = best_version(x_train, y_train, x_val, y_val, knn, score, grid, -1)
    knn = KNeighborsClassifier(**best.best_params_)

    knn.fit(X_train, Y_train)

    # Work on test data and predict
    loader = DataLoader(test, batch_size=args["batch_size"], shuffle=False, num_workers=2, drop_last=False)

    predictions, label_list = [], []
    for images, labels in loader:
        images = images.to(args["device"])
        label_list += labels
        with torch.no_grad():
            outputs = model(images, features=True)
            for output in outputs:
                prediction = knn.predict([np.array(output.cpu())])
                predictions.append(prediction)

    accuracy = accuracy_score(label_list, predictions)

    return accuracy, predictions, label_list


def best_version(x_train, y_train, x_val, y_val, regressor, score, grid, njobs):
    gs = GridSearchCV(regressor, grid, cv=3, scoring=score, n_jobs=njobs, verbose=1)
    gs.fit(x_train, y_train)
    print("Best parameters set:")
    print(gs.best_params_)
    y_pred = gs.predict(x_val)
    print(f"accuracy score: {accuracy_score(y_val,y_pred)}")
    return gs

def CELoss(outputs,targets):
    logsoftmax = torch.nn.LogSoftmax()
    softmax = torch.nn.Softmax()
    return torch.mean(torch.sum(- softmax(targets) * logsoftmax(outputs),1))

def accuracyPlot(accuracies, std, names, mytitle):

    fig = go.Figure()
    for idx, el in enumerate(names):
        fig.add_trace(go.Scatter(
            ####101 10
            x=list(np.arange(10, 51, 10)),
            y=accuracies[idx],
            error_y=dict(
                type='data',
                array=std[idx]
            ),
            name=el
        ))

    #####
    array = list(np.around(np.arange(0, 1.1, 0.1), decimals=1))
    for i in array:
        fig.add_shape(
            dict(
                type="line",
                x0=0,
                y0=i,
                #100
                x1=50,
                y1=i,
                line=dict(
                    color="Grey",
                    width=1,
                    dash="dot",
                )
            ))
    ####
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1], dtick=0.1, tickcolor='black', ticks="outside",
                                  tickwidth=1, ticklen=5)
    fig['layout']['xaxis'].update(title='Number of classes', range=[0, 50.5], dtick=10, ticks="outside", tickwidth=0)
    fig['layout'].update(height=700, width=900)
    fig['layout'].update(plot_bgcolor='rgb(256,256,256)')
    if mytitle!='':
        fig.update_layout(title=mytitle)
    fig.show()
    return fig

  
def avg(seed1, seed2, seed3):
    avg=[]
    std=[]
    for i in range(len(seed1)):
        s=[seed1[i], seed2[i], seed3[i]]
        avg.append(statistics.mean(s))
        std.append(statistics.stdev(s))
    return avg, std
    
def harmonicMean(closedW, closedStd, openW, openStd):
    harmonic_mean=[]
    std=[]
    for i in range(len(closedW)):
        harmonic_mean.append((2 * closedW[i] * openW[i]) / (closedW[i] + openW[i]))
        std.append((closedStd[i] + openStd[i])/2)
    return harmonic_mean, std
