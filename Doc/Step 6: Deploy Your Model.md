# 第6步：部署模型

您可以在Google Cloud上训练，调整和部署机器学习模型。 有关将模型部署到生产的指导，请参阅以下资源：

*  有关如何使用TensorFlow服务导出Keras模型的[教程](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#exporting-a-model-with-tensorflow-serving)。
*  TensorFlow服务文档。
*  在Google Cloud上培训和部署模型的指南。
*  部署模型时请记住以下关键事项：

*  确保您的生产数据遵循与培训和评估数据相同的分布。
*  通过收集更多培训数据定期重新评估。
*  如果您的数据分布发生变化，请重新训练模型。
