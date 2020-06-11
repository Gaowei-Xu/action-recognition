#!/usr/bin/python3
# -*-coding:utf-8-*-


import os
import json

root_dir = '../dataset/'
train_val_txt_path='../trainval.txt'

annotation_files = [f for f in os.listdir(root_dir) if f.endswith('.json') and 'status' not in f]
old_pattern = '郑思维-黄雅琼vs德差波尔-沙西丽'
new_pattern = 'zsw-hyq-vs-debe-sxi'

samples = list()
for f_name in annotation_files:
    # replace chinese characters with given string

    labels_info = json.load(open(os.path.join(root_dir, f_name)))
    for item in labels_info:
        video_name = item['video']
        label = item['label']

        video_full_path = os.path.join(
            video_name.split('\\')[0],
            video_name.split('\\')[1])

        video_full_path = video_full_path.replace(old_pattern, new_pattern)

        if not os.path.exists(os.path.join(root_dir, video_full_path)):
            continue

        samples.append(dict(
            video_full_path=video_full_path,
            label=label
        ))


print('Total samples amount = {}'.format(len(samples)))



labels = list(set([s['label'] for s in samples]))
mapping = dict()
for i, label in enumerate(labels):
    mapping[label] = i

with open(train_val_txt_path, 'w', encoding='utf-8') as wf:
    for sample in samples:
        wf.write('{} {} {}\n'.format(
            sample['video_full_path'],
            100,
            mapping[sample['label']]
        ))

json.dump(mapping, open('../mapping.json', 'w'), ensure_ascii=False, indent=True)

# Step 3: custom's network
# Dependency: pip install gluoncv

# ! pip install gluoncv
# ! pip install python-opencv
# !pip install torch
import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory

num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
transform_train = video.VideoGroupTrainTransform(
    size=(224, 224),
    scale_ratios=[1.0, 0.8],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

per_device_batch_size = 5
num_workers = 8
batch_size = per_device_batch_size * num_gpus

train_dataset = VideoClsCustom(
    root=os.path.expanduser(root_dir),
    setting=os.path.expanduser(train_val_txt_path),
    train=True,
    new_length=32,
    video_loader=True,
    transform=transform_train)

print('Load %d training samples.' % len(train_dataset))
train_data = gluon.data.DataLoader(
    train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=num_workers)

net = get_model(name='slowfast_8x8_resnet50_kinetics400', nclass=18)
net.collect_params().reset_ctx(ctx)
# print(net)

# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [40, 80, 100]

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': 0.001, 'wd': 0.0001, 'momentum': 0.9}

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-acc'])


epochs = 5
lr_decay_count = 0

for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0

    # Learning rate decay
    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        print('Batch {}...'.format(i))
        # Extract data and label
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        # AutoGrad
        with ag.record():
            output = []
            for _, X in enumerate(data):
                X = X.reshape((-1,) + X.shape[2:])
                pred = net(X)
                output.append(pred)
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size)

        # Update metrics
        train_loss += sum([l.mean().asscalar() for l in loss])
        train_metric.update(label, output)

        if i == 100:
            break

    name, acc = train_metric.get()

    # Update history and print metrics
    train_history.update([acc])
    print('[Epoch %d] train=%f loss=%f time: %f' %(epoch, acc, train_loss / (i+1), time.time()-tic))

# We can plot the metric scores with:
train_history.plot()


