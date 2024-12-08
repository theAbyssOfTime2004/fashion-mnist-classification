# FASHION MNIST Classification Using Machine Learning (SVM Model)

## Data introduction 

- Training data: 60,000 grayscale 28x28 pixel images, converted to 784-element 1D arrays.
- Labels: 10 categories including T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- Test data: 10,000 images with corresponding labels, structured similarly to training data.
  
## Data Loading And Preprocessing
- The data loading and preprocessing section focuses on preparing the MNIST dataset for further analysis. It handles extracting image and label data from compressed files and organizing them into suitable formats for machine learning tasks. The images are flattened into vectors, each representing a single image as a series of pixel intensity values, while the labels provide corresponding class information for these images. This step ensures the dataset is properly structured for subsequent modeling and analysis.
  
## Model Building

### Linear SVM Model: 
- A Linear Support Vector Classifier (LinearSVC) is initialized with a fixed random state for reproducibility.
- The model is trained on the MNIST training data.
- Accuracy is calculated on the test dataset, and predictions are made to assess the classifier's performance.
- A classification report is generated to show precision, recall, and F1-score for each class.
  
### RBF kernel SVM (Nystroem Approximation):
- An RBF kernel approximation is implemented using the `Nystroem` transformer, reducing the computational cost of the kernel method.
- A pipeline is constructed to integrate the RBF kernel approximation with a LinearSVC classifier.
- The pipeline is trained on the training data and evaluated on the test set.
- The pipeline's accuracy and classification report are calculated similarly to the Linear SVM.

The script provides a practical example of the "curse of dimensionality" in machine learning. As demonstrated, high-dimensional data (such as the original 784-feature MNIST dataset) can complicate model training, affecting both computational efficiency and prediction accuracy. Dimensionality reduction techniques like PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis) are employed to mitigate this issue by reducing the data to a smaller set of features.
## Dimensionality Reduction with PCA and LDA
Apply dimensionality reduction techniques to enhance computational efficiency and compares their impact on the performance of Linear SVM and RBF kernel models.

### Dimensionality Reduction on Linear SVM
- PCA (Principal Component Analysis):
  - PCA reduces the dataset's dimensionality to 100 components while preserving as much variance as possible.
  - The cumulative explained variance ratio is analyzed.
  - The transformed dataset is used to train and evaluate the LinearSVC model.
  - Training and transformation times are combined, and prediction accuracy is calculated.
- LDA (Linear Discriminant Analysis):
  - LDA reduces dimensionality to 9 components based on the number of classes in the data.
  - It maximizes class separability by creating discriminative features.
  - The transformed dataset is used to train and evaluate the LinearSVC model.
  - Training and transformation times are combined, and accuracy is recorded.
    
### Dimensionality Reduction on RBF Kernel (Nystroem Approximation)
- PCA:
  - PCA reduces dimensionality to 100 components before applying the RBF kernel with Nystroem approximation.
  - The pipeline is trained and evaluated, with training, transformation, and prediction times recorded.
- LDA:
  - LDA reduces dimensionality to 9 components before applying the RBF kernel with Nystroem approximation.
  - Training, transformation, and prediction times are recorded.
    
### Analysis:
- Trade-offs:
 - PCA significantly reduces the dimensionality, making computations faster while retaining most of the data's variance. LDA optimizes for class separability but may be less effective for datasets with high variability within classes.
 - Using dimensionality reduction generally reduces training time but may slightly impact accuracy.
- Model-Specific Performance:
 - LinearSVC benefits from reduced dimensionality with PCA and LDA, leading to faster computations.
 - RBF Kernel models see improved efficiency with dimensionality reduction while maintaining competitive accuracy.
- Insights on PCA and LDA:
 - PCA retains ~XX% of variance with 100 components (actual value provided in the detailed analysis).
 - LDA is constrained by the number of classes, making it a targeted but limited reduction approach.
   
## Key Observations and Insights:
1. **Training and Prediction Time**: Reducing dimensions with PCA and LDA leads to faster training and prediction times compared to using the full 784 features. This confirms that as dimensionality increases, computational requirements grow substantially, resulting in longer processing times. By reducing the number of features, models can operate more efficiently without necessarily compromising accuracy.

2. **Model Accuracy**: Despite the reduction in feature space, the model's accuracy does not degrade significantly when using PCA or LDA. In fact, PCA with 100 components captures a significant amount of variance, retaining critical information while reducing dimensionality. LDA, which is more sensitive to class separability, also provides an effective feature reduction strategy by focusing on dimensions that maximize class distinctions.

3. **Effectiveness of Kernel Approximation**: Using kernel approximation (Nystroem) with the RBF kernel in combination with dimensionality reduction shows that even complex models can perform well with fewer features. This further supports the idea that high-dimensional data can often be reduced to a more manageable number of features without a significant loss in performance.

## Conclusion
The script illustrates that the curse of dimensionality can indeed be alleviated through dimensionality reduction techniques like PCA and LDA. High-dimensional data can lead to increased computational costs and, in some cases, decreased model performance due to overfitting or sparsity issues. Reducing the dimensionality allows for faster processing, often without substantial loss in accuracy, underscoring the importance of selecting an appropriate number of dimensions to balance efficiency and predictive power.
