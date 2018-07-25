# 附录：批量训练

非常大的数据集可能不适合分配给您的进程的内存。 在前面的步骤中，我们设置了一个管道，我们将整个数据集引入内存，准备数据，并将工作集传递给培训功能。 相反，Keras提供了另一种培训功能（fit_generator），可以批量提取数据。 这允许我们将数据管道中的转换仅应用于数据的一小部分（batch_size的一部分）。 在我们的实验中，我们使用批处理（GitHub中的代码）来处理数据集，例如DBPedia，Amazon评论，Ag新闻和Yelp评论。

以下代码说明了如何生成数据批并将其提供给fit_generator。
```python
def _data_generator(x, y, num_features, batch_size):
    """Generates batches of vectorized texts for training/validation.

    # Arguments
        x: np.matrix, feature matrix.
        y: np.ndarray, labels.
        num_features: int, number of features.
        batch_size: int, number of samples per batch.

    # Returns
        Yields feature and label data in batches.
    """
    num_samples = x.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    while 1:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_samples:
                end_idx = num_samples
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            yield x_batch, y_batch

# Create training and validation generators.
training_generator = _data_generator(
    x_train, train_labels, num_features, batch_size)
validation_generator = _data_generator(
    x_val, val_labels, num_features, batch_size)

# Get number of training steps. This indicated the number of steps it takes
# to cover all samples in one epoch.
steps_per_epoch = x_train.shape[0] // batch_size
if x_train.shape[0] % batch_size:
    steps_per_epoch += 1

# Get number of validation steps.
validation_steps = x_val.shape[0] // batch_size
if x_val.shape[0] % batch_size:
    validation_steps += 1

# Train and validate model.
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    epochs=epochs,
    verbose=2)  # Logs once per epoch.

```
