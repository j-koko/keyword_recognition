## **Final Report: Keyword Recognition with CNN and Convolutional Autoencoder**

**Authors:** Stella Siu, Jan Kokowski, Tiantian Zhang

### **Task I: Improving the CNN Model**

#### **Baseline Performance**

The base CNN model achieved a validation accuracy of 85.29% after 10 epochs. The task involved understanding the architecture and exploring methods to optimize the model for improved performance.

#### **Experiment 1: Increasing the Number of Epochs**

Our first attempt was to increase the number of training epochs. Disabling early stopping allowed the model to train for up to 50 epochs. However, this resulted in high fluctuations in validation accuracy and eventual overfitting, as the model failed to generalize to unseen data.

#### **Experiment 2: Resizing Input Layer**

To explore further improvements, we resized the input spectrograms to 64x64. This was based on the hypothesis that preserving fine-grained details in spectrograms could help convolutional layers extract richer features and improve classification accuracy. Resizing combined with training for 20 epochs yielded optimal performance, achieving the following results on the test set:

* **Accuracy**: 90.38%  
* **Loss**: 0.33

In comparison, the base model achieved:

* **Accuracy**: 83.77%  
* **Loss**: 0.47

Resizing the spectrograms allowed the convolutional layers to better capture spatial dependencies, contributing to improved generalization.

#### **Experiment 3: Global Pooling Layer**

We experimented with replacing the final pooling layer with a global pooling layer, hypothesizing it might improve generalization. However, this approach significantly underperformed, yielding a validation accuracy of only 48.05%. We believe the poor results were due to the loss of spatial information critical for spectrogram-based recognition tasks, such as formant structures. This experiment demonstrated that preserving spatial dependencies is essential for tasks involving spectrograms.

#### **Experiment 4: Reducing Batch Size**

Another experiment involved reducing the batch size while increasing the number of epochs. While this resulted in moderate improvements, the approach caused excessive fluctuations in both validation loss and accuracy, often triggering early stopping. Due to its instability, we abandoned this approach.

---

### **Task II: Replacing CNN with a Convolutional Autoencoder**

#### **Model Architecture**

For the second task, we replaced the CNN with a convolutional autoencoder. The architecture consisted of two main components:

1. **Encoder**:  
   * Processes input spectrograms of dimensions `(124, 129, 1)` through a series of convolutional layers.  
   * Each convolutional layer is followed by batch normalization to stabilize learning, ReLU activation to introduce non-linearity, and max pooling to downsample spatial dimensions while retaining critical features.  
   * The output feature maps are flattened into a compact latent representation of size 34,816.  
2. **Classifier**:  
   * A fully connected dense layer with softmax activation to classify the 10 keywords.

We avoided global pooling in the encoder based on its poor performance in Task I.

#### **Training and Performance**

The autoencoder achieved decent performance after 50 epochs. The model reached a training accuracy of 99.50%, indicating strong learning capability, but the gap between training and validation metrics suggested overfitting. Despite the overfitting, the validation accuracy reached 87.37%, demonstrating good generalization. We noticed noticeable fluctuations in validation accuracy and loss, likely due to the small validation set of 1,600 recordings.

#### **Regularization Attempts**

We hypothesized that the fluctuations might result from insufficient regularization. However, adding dropout destabilized the model further, likely because the small dataset made the model sensitive to stochastic regularization. This indicated that improving stability may require careful adjustments to the regularization strategy or expanding the dataset.

---

### **Results and Analysis**

#### **Performance Comparison**

The autoencoder's final accuracy was:

* **Accuracy**: 85.46%  
* **Loss**: 1.02

The CNN improved with resizing and more epochs achieved slightly better results:

* **Accuracy**: 90.38%  
* **Loss**: 0.33

#### **Confusion Matrix Analysis**

The confusion matrix revealed that the autoencoder accurately recognized most keywords. However, it struggled with acoustically similar words or those sharing phonemes, such as "go" and "no" (7 misclassifications) and "go" and "down" (15 misclassifications). These errors highlight challenges in distinguishing acoustically overlapping patterns.

#### **Key Takeaways**

* The CNN achieved slightly better performance due to its simpler architecture and reduced overfitting.  
* The autoencoder demonstrated the potential to model complex latent patterns and provided comparable performance despite the challenges of a small dataset.

---

### **Conclusion and Future Work**

The improved CNN and convolutional autoencoder both performed well on the keyword recognition task. While the CNN achieved higher accuracy, the autoencoder showcased its ability to effectively extract latent features for classification. Future improvements could include:

1. Expanding the validation set to stabilize performance and reduce fluctuations.  
2. Exploring advanced regularization strategies, such as weight decay or dropout tuning, to improve generalization.  
3. Experimenting with hybrid architectures combining the strengths of CNNs and autoencoders.

