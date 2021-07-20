import math
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from data import augmentate
from utils import FCClassifier, nme_classifier, svm_classifier, confusionMatrix, get_oneHot_labels, knn_classifier
# Lwf (Learning without forgetting)


def run_lwf(classif_loss, distill_loss, train_batches, test_batches, model, finetuning, args):
    '''
    Run method for Learning Without Forgetting

    Parameters:
        criterion: loss function
        train_batches: batches used for training
        test_batches: batches used for testing
        model: the resnet used
        args: dict containing the static variables

    '''
    accuracy_batch = []
    for idx, batch in enumerate(train_batches):
        n_classes = (idx+1)*10
        model = update_representation(model, batch, {}, classif_loss, distill_loss, n_classes, args, finetuning)
        accuracy, predictions, labels = FCClassifier(test_batches[idx], model, n_classes, args["device"])
        accuracy_batch.append(accuracy)
        confusionMatrix(labels, predictions, idx)

    return accuracy_batch

# Algorithm 2 (Incremental Train)


def run(train_batches, test_batches, model, classifier, classif_loss, distill_loss, args, finetuning, mode):
    '''
    Run method of each version of iCarL depending on the input parameters

    Parameters:
        train_batches: batches used for training
        test_batches: batches used for testing
        model: the resnet used
        classifier: string, can be NME for nearest mean classifier, KNN for KNN classifier, SVM for SVM classifier and FC for fully connected
        classif_loss: loss function used for classification
        distill_loss: loss function used for distillation
        args: dict containing the static variables
        mode: string, can be default for default iCarL's paper, cwwr for closed world with rejection, ownr owr open world with naive rejection or ownr for open world without naive rejection
    '''
    accuracy_batch = []
    exemplars = {}
    print(f"Running in mode: {mode}")
    for idx, batch in enumerate(train_batches):
        print(f'\n Batch #{idx+1}')

        # Define new number of classes (incremented of 10)
        n_class_incr = (idx+1)*10
        # classes_group=n_class_incr-k # double check with others

        # Algorithm 3 (Update Representation)
        print('\n Update Representation: ')
        model = update_representation(model, batch, exemplars, classif_loss, distill_loss, n_class_incr, args, finetuning)

        # Algorithm 4 (Construct Exemplars) --> Consider taking a random set of exemplars, herding is long process
        new_exemplars = construct_exemplars_set(model, batch, n_class_incr, args)

        # Update exemplars dictionary
        exemplars.update(new_exemplars)

        # Classification
        if mode == "default":
            if classifier == 'FC':
                accuracy, predictions, labels = FCClassifier(test_batches[idx], model, n_class_incr, args["device"])
            elif classifier == 'NME':
                accuracy, predictions, labels = nme_classifier(test_batches[idx], batch, exemplars, model, n_class_incr, args, mode)
            elif classifier == 'SVM':
                accuracy, predictions, labels = svm_classifier(exemplars, test_batches[idx], model, args)
            elif classifier == 'KNN':
                accuracy, predictions, labels = knn_classifier(exemplars, test_batches[idx], model, args)
        else:
            if idx + 1 >= 5:
                break
            if mode == "cwwr":  # closed world with naive rejection
                if classifier in ('FC', 'SVM', 'KNN'):
                    raise NotImplementedError
                elif classifier == 'NME':
                   
                    accuracy, predictions, labels = nme_classifier(test_batches[idx], batch, exemplars, model, n_class_incr, args, mode)
            elif mode in ("owr", "ownr"):  # open world
                if classifier in ('FC', 'SVM', 'KNN'):
                    raise NotImplementedError
                elif classifier == 'NME':
                    accuracy, predictions, labels = nme_classifier(test_batches[idx+5], batch, exemplars, model, n_class_incr, args, mode)


        # Add accuracy of batch
        accuracy_batch.append(accuracy)
        print(f'\n Accuracy NME Batch #{idx+1}: {accuracy}')

        # Plot confusion matrix --> è già in utils
        if mode == "default" or mode == "cwwr":
            confusionMatrix(labels, predictions, idx)

        exemplars = reduce_exemplars_set(exemplars, n_class_incr, args["memory"])

    return accuracy_batch

# Algorithm 3 (Update Representation)


def update_representation(model, data, exemplars, classif_loss, distill_loss, n_classes, args, finetuning):
    '''
    Algorithm 3 in iCarL's paper

    Parameters:
        model: the resnet used
        data:training data
        exemplars: set of exemplars
        classif_loss: loss function used for classification
        distill_loss: loss function used for distillation
        n_classes: total number of classes
        args: dictionary containing the static values
    '''
    epochs: int = 70
    model.to(args["device"])

    if len(exemplars) != 0:
        new_exemplars = []
        for key in exemplars:
            for item in exemplars[key]:
                new_exemplars.append([item[0], item[1]])
        data += new_exemplars
    loader = DataLoader(data, args["batch_size"], shuffle=True, num_workers=2, drop_last=False)
    # changed classes_group to n classes here and in torch.nn.Linear

    if n_classes != 10:
        # Save network for distillation
        old_model = deepcopy(model)
        old_model.eval()
        # Update network's last layer
        _input = model.linear.in_features
        output = model.linear.out_features
        weight = model.linear.weight.data
        bias = model.linear.bias.data

        model.linear = torch.nn.Linear(_input, n_classes)
        model.linear.weight.data[:output] = weight
        model.linear.bias.data[:output] = bias

    model = model.to(args["device"])
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['wd'], momentum=args['mom'])

    for epoch in range(epochs):
        if epoch == 49 or epoch == 63:
            optimizer.param_groups[0]['lr'] /= 5
        # Set module in training mode
        model.train()

        running_loss = 0
        for images, labels in loader:
            images = images.to(args["device"])
            images = torch.stack([augmentate(image) for image in images])

            # Zero-ing the gradients
            optimizer.zero_grad()
            # Forward pass to the network
            # Get One Hot Encoding for the labels
            if 'BCE' in str(classif_loss):
                outputs = model(images)
            if 'MSE' in str(classif_loss):
                outputs = torch.sigmoid(model(images))

            labels = get_oneHot_labels(labels, n_classes)
            labels = labels.to(args["device"])

            # Compute Losses
            if n_classes == 10 or finetuning:
                tot_loss = classif_loss(outputs, labels)
            else:
                with torch.no_grad():
                    old_outputs = torch.sigmoid(old_model(images))
                if 'BCE' in str(distill_loss):
                    targets = torch.cat((old_outputs, labels[:, n_classes-10:]), 1)
                    tot_loss = distill_loss(outputs, targets)
                elif 'MSE' in str(distill_loss):
                    class_loss = classif_loss(outputs, labels)
                    dist_loss = torch.pow(classif_loss(torch.pow(outputs[:,:n_classes-10],2),torch.pow(old_outputs,2)), 1/2)
                    tot_loss = dist_loss*.2 + class_loss
            

            # Update Running Loss
            running_loss += tot_loss.item() * images.size(0)

            tot_loss.backward()
            optimizer.step()

        # Train loss of current epoch
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} loss: {running_loss/len(data):.4f}, lr: {optimizer.param_groups[0]['lr']}")
    print()
    return model

# Algorithm 4 (Construct Exemplars)


def construct_exemplars_set(model, data, n_classes, args):
    '''
    Algorithm 4 in iCarL's paper

    Parameters:
        model: the resnet used
        data: the training data on which to build the exemplars
        n_classes: total number of classes
        args: dictionary containing the static values
    '''
    # We need to get the mean of the images for the set of images in a class
    # Then we need to stack them or otherwise put them together and pop/remove the worst ones untile we reach n_class - n class_batch
    print("Constructing examplar set...")
    m: int = math.floor(args["memory"]/n_classes)
    classes: range = range(n_classes-10, n_classes)
    modified_data: dict = {el: [] for el in classes}
    print("Building images/labels map...", end="\r")

    for label in classes:
        modified_data[label] = []
        # Fill class_map
        for item in data:
            if item[1] == label:
                modified_data[label].append(item)
    print("Images/label map created. Creating dictionary with features of each class...")
    model.eval()  # We don't want train just evaulate the classes
    outputs_class = {el: [] for el in classes}
    means: dict = {el: torch.zeros((1, 64), device=args["device"]) for el in classes}
    exemplars: dict = {el: [] for el in classes}
    for key in modified_data:
        with torch.no_grad():
            class_dataloader = DataLoader(modified_data[label], batch_size=args["batch_size"], shuffle=False, drop_last=False)
            for images, _ in class_dataloader:
                images = images.to(args["device"])
                outputs = model(images, features=True)  # outputs features instead of labels (1x64 instead of 32x32 but it doesn't matter here)
                for output in outputs:
                    means[key] += output
                # print(len(outputs))
                outputs_class[key].extend(outputs)

        means[key] /= len(outputs_class[key])
        means[key] /= means[key].norm()

    print("Feature extracted for every class. Starting build of exemplars sets for each class...")

    for label in classes:
        choosen_features = []
        choosen_ids = list()
        w_t = means[label].squeeze()  # perché .squeeze() -> capire perché è shape 1,64 al posto di 64
        for k in range(0, min(m, len((outputs_class[label])))):
            # diff = 1e15
            maximum = -1e15
            # min_idx = None
            ind_max = None

            for idx, phi_x in enumerate(outputs_class[label]):
                if idx in choosen_ids:
                    continue

                dot = w_t.dot(phi_x)

                if dot > maximum:
                    maximum = dot
                    ind_max = idx

            w_t = w_t+means[label].squeeze()-outputs_class[label][ind_max]
            choosen_ids.append(ind_max)
            choosen_features.append(outputs_class[label][ind_max])
            exemplars[label].append(modified_data[label][ind_max])
        print(f"Exemplar set for class {label} created with dimension: {len(exemplars[label])}. Starting next class...")
    print("Set of examplars built.")

    return exemplars

# Algorithm 5 (Reduce exemplar set)


def reduce_exemplars_set(exemplars, classes_n, memory):
    '''
    Algorithm 5 in iCarL's paper, reduces the size of the exemplars' set

    Parameters:
        exemplars: set of exemplars
        classes_n: total class number
        memory: total max number of exemplars
    '''
    memory_per_class = math.floor(memory/classes_n)
    for key in exemplars:
        exemplars[key] = exemplars[key][:memory_per_class]
    return exemplars
