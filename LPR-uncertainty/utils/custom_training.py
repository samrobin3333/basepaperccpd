"""Part of the training engine related to Python generators of array data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import copy
import numpy as np


import tensorflow as tf
import keras
from keras import backend as K
from keras.utils.generic_utils import slice_arrays
from keras.utils.generic_utils import to_list
import numpy as np
import tensorflow as tf
from scipy.sparse import issparse

from keras.engine.training_utils import batch_shuffle
from keras.engine.training_utils import make_batches
from keras.engine.training_utils import check_num_samples
from keras import callbacks as cbks
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import slice_arrays
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton


def fit(model,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            custom_callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
    # Backwards compatibility
    if batch_size is None and steps_per_epoch is None:
        batch_size = 32
    # Legacy support
    if 'nb_epoch' in kwargs:
        warnings.warn('The `nb_epoch` argument in `fit` '
                      'has been renamed `epochs`.', stacklevel=2)
        epochs = kwargs.pop('nb_epoch')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
    if x is None and y is None and steps_per_epoch is None:
        raise ValueError('If fitting from data tensors, '
                         'you should specify the `steps_per_epoch` '
                         'argument.')
    # Validate user data.
    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        batch_size=batch_size)
    # Prepare validation data.
    do_validation = False
    if validation_data:
        do_validation = True
        if len(validation_data) == 2:
            val_x, val_y = validation_data
            val_sample_weight = None
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data
        else:
            raise ValueError('When passing validation_data, '
                             'it must contain 2 (x_val, y_val) '
                             'or 3 (x_val, y_val, val_sample_weights) '
                             'items, however it contains %d items' %
                             len(validation_data))

        val_x, val_y, val_sample_weights = model._standardize_user_data(
            val_x, val_y,
            sample_weight=val_sample_weight,
            batch_size=batch_size)
        if model._uses_dynamic_learning_phase():
            val_ins = val_x + val_y + val_sample_weights + [0.]
        else:
            val_ins = val_x + val_y + val_sample_weights

    elif validation_split and 0. < validation_split < 1.:
        if any(K.is_tensor(t) for t in x):
            raise ValueError(
                'If your data is in the form of symbolic tensors, '
                'you cannot use `validation_split`.')
        do_validation = True
        if hasattr(x[0], 'shape'):
            split_at = int(int(x[0].shape[0]) * (1. - validation_split))
        else:
            split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_arrays(x, 0, split_at),
                    slice_arrays(x, split_at))
        y, val_y = (slice_arrays(y, 0, split_at),
                    slice_arrays(y, split_at))
        sample_weights, val_sample_weights = (
            slice_arrays(sample_weights, 0, split_at),
            slice_arrays(sample_weights, split_at))
        if model._uses_dynamic_learning_phase():
            val_ins = val_x + val_y + val_sample_weights + [0.]
        else:
            val_ins = val_x + val_y + val_sample_weights

    elif validation_steps:
        do_validation = True
        if model._uses_dynamic_learning_phase():
            val_ins = [0.]

    # Prepare input arrays and training function.
    if model._uses_dynamic_learning_phase():
        ins = x + y + sample_weights + [1.]
    else:
        ins = x + y + sample_weights
    model._make_train_function()
    f = model.train_function

    # Prepare display labels.
    out_labels = model.metrics_names

    if do_validation:
        model._make_test_function()
        val_f = model.test_function
        callback_metrics = copy.copy(out_labels) + [
            'val_' + n for n in out_labels]
    else:
        callback_metrics = copy.copy(out_labels)
        val_f = None
        val_ins = []

    return fit_loop(model, f, ins,
                        out_labels=out_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        custom_callbacks=custom_callbacks,
                        val_f=val_f,
                        val_ins=val_ins,
                        shuffle=shuffle,
                        callback_metrics=callback_metrics,
                        initial_epoch=initial_epoch,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)


def fit_loop(model, f, ins,
             out_labels=None,
             batch_size=None,
             epochs=100,
             verbose=1,
             callbacks=None,
             custom_callbacks=None,
             val_f=None,
             val_ins=None,
             shuffle=True,
             callback_metrics=None,
             initial_epoch=0,
             steps_per_epoch=None,
             validation_steps=None):

    do_validation = False
    if val_f and val_ins:
        do_validation = True
        if (verbose and ins and
           hasattr(ins[0], 'shape') and hasattr(val_ins[0], 'shape')):
            print('Train on %d samples, validate on %d samples' %
                  (ins[0].shape[0], val_ins[0].shape[0]))
    if validation_steps:
        do_validation = True
        if steps_per_epoch is None:
            raise ValueError('Can only use `validation_steps` '
                             'when doing step-wise '
                             'training, i.e. `steps_per_epoch` '
                             'must be set.')
    elif do_validation:
        if steps_per_epoch:
            raise ValueError('Must specify `validation_steps` '
                             'to perform validation '
                             'when doing step-wise training.')

    num_train_samples = check_num_samples(ins,
                                          batch_size=batch_size,
                                          steps=steps_per_epoch,
                                          steps_name='steps_per_epoch')
    if num_train_samples is not None:
        index_array = np.arange(num_train_samples)

    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    if verbose:
        if steps_per_epoch is not None:
            count_mode = 'steps'
        else:
            count_mode = 'samples'
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode,
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)
    out_labels = out_labels or []

    # it's possible to callback a different model than itself
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model

    callbacks.set_model(callback_model)
    custom_callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    custom_callbacks.custom_on_train_begin()
    callback_model.stop_training = False
    for cbk in callbacks:
        cbk.validation_data = val_ins

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(ins[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    for epoch in range(initial_epoch, epochs):
        # Reset stateful metrics
        for m in model.stateful_metric_functions:
            m.reset_states()
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        if steps_per_epoch is not None:
            for step_index in range(steps_per_epoch):
                batch_logs = {}
                batch_logs['batch'] = step_index
                batch_logs['size'] = 1
                callbacks.on_batch_begin(step_index, batch_logs)
                outs = f(ins)

                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                #custom_callbacks.custom_on_batch_end(x, y, outs, batch_index, batch_logs)
                callbacks.on_batch_end(step_index, batch_logs)
                if callback_model.stop_training:
                    break

            if do_validation:
                val_outs = test_loop(model, val_f, val_ins,
                                     steps=validation_steps,
                                     verbose=0)
                val_outs = to_list(val_outs)
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o
        else:
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(num_train_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(ins[-1], float):
                        # Do not slice the training phase flag.
                        ins_batch = slice_arrays(
                            ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = slice_arrays(ins, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                ins_batch[0] = custom_callbacks.custom_on_batch_begin(x=ins_batch[0], y=[ins_batch[1], ins_batch[2]])
                for i in indices_for_conversion_to_dense:
                    ins_batch[i] = ins_batch[i].toarray()

                #with tf.GradientTape(watch_accessed_variables=False) as g:
                    #g.watch(model.trainable_weights[4*6])
                outs = f(ins_batch)
                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o


                custom_callbacks.custom_on_batch_end(x=ins_batch[0], y=[ins_batch[1],ins_batch[2]], outs=outs, batch=batch_index, logs=batch_logs)
                callbacks.on_batch_end(batch_index, batch_logs)
                if callback_model.stop_training:
                    break

                if batch_index == len(batches) - 1:  # Last batch.
                    if do_validation:
                        val_outs = test_loop(model, val_f, val_ins,
                                             batch_size=batch_size,
                                             verbose=0)
                        val_outs = to_list(val_outs)
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

        custom_callbacks.custom_on_epoch_end(x=ins_batch[0], y=[ins_batch[1], ins_batch[2]], outs=outs,
                                             epoch=epoch, logs=batch_logs)
        if model.loss_weights is not None:
            epoch_logs['sr_weight'] = K.get_value(model.loss_weights[0])
            epoch_logs['cl_weight'] = K.get_value(model.loss_weights[1])

        callbacks.on_epoch_end(epoch, epoch_logs)
        if callback_model.stop_training:
            break
    callbacks.on_train_end()
    return model.history


def test_loop(model, f, ins, batch_size=None, verbose=0, steps=None):


    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    num_samples = check_num_samples(ins,
                                    batch_size=batch_size,
                                    steps=steps,
                                    steps_name='steps')
    outs = []
    if verbose == 1:
        if steps is not None:
            progbar = Progbar(target=steps)
        else:
            progbar = Progbar(target=num_samples)

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(ins[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    if steps is not None:
        for step in range(steps):
            batch_outs = f(ins)
            if isinstance(batch_outs, list):
                if step == 0:
                    for _ in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    if i in stateful_metric_indices:
                        outs[i] = float(batch_out)
                    else:
                        outs[i] += batch_out
            else:
                if step == 0:
                    outs.append(0.)
                outs[0] += batch_outs
            if verbose == 1:
                progbar.update(step + 1)
        for i in range(len(outs)):
            if i not in stateful_metric_indices:
                outs[i] /= steps
    else:
        batches = make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if isinstance(ins[-1], float):
                # Do not slice the training phase flag.
                ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_arrays(ins, batch_ids)
            for i in indices_for_conversion_to_dense:
                ins_batch[i] = ins_batch[i].toarray()

            batch_outs = f(ins_batch)
            if isinstance(batch_outs, list):
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    if i in stateful_metric_indices:
                        outs[i] = batch_out
                    else:
                        outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i in range(len(outs)):
            if i not in stateful_metric_indices:
                outs[i] /= num_samples
    return unpack_singleton(outs)