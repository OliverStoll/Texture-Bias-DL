# On the Prevalence of Texture Bias in Deep Learning Models for Remote Sensing

This project investigates the phenomenon of texture bias in deep learning models, particularly Convolutional Neural Networks (CNNs) and Vision Transformers. 
It provides tools to analyze and visualize, feature bias across various datasets and model architectures, to gain an understanding which image features models depend on for classification of specific datasets.

It was created as part of my [Master Thesis](Thesis.pdf) _"On the Prevalence of Texture Bias in Deep Learning Models for Remote Sensing"_.

## 📖 Background

Recent studies have highlighted that CNNs trained on datasets like ImageNet tend to prioritize texture over shape when classifying images, contrasting with human visual perception. 
This texture bias can lead to reduced robustness and generalization in models. Understanding and addressing this bias is crucial for developing more reliable computer vision systems.

However, this project finds that CNNs are in fact not overly reliant on texture features, but on shape features as previously hypothesized.

## 🔍 Project Overview

This project offers a suite of tools and scripts to:

- Analyze classification performance of models under various transformations, both overall or class-wise for each dataset.
- Visualize the impact of feature bias such as texture, shape or color on model predictions.
- Predict which image features models are most reliant for correct classification of specific datasets.

## 🗂️ Repository Structure

The repository is organized as follows:

- `src/`: Entire Pytorch Lightning based codebase of the project
  - `data_loading/`: Contains dataloaders and preprocessing scripts for integrated datasets.
  - `models/`: Model factory for easy instantiation of various `timm` models.
  - `training/`: Training scripts and utilities.
  - `transforms/`: Data augmentation and transformation scripts.
  - `checks/`: Sanity checks to ensure correct data loading and model training.
  - `evaluation/`: Plotting of model performance under various feature suppression transformations.
- `data/`: Directory for storing datasets and related files, as well as example data for testing. Additionally it includes results on the comparison of feature importance on multiple CV and RS datasets related to my thesis.
- `figures/`: Contains generated figures and visualizations.
- `config.yml`: Configuration file for setting up experiments and parameters.

## 📊 Key Features

- **Custom Dataset & Model Support**: Evaluate the feature bias of any `timm` model on any dataset
- **Class-wise Analysis**: Evaluate feature importance via model performances on individual classes of each dataset.
- **Visualization Tools**: Generate plots to visualize the extent of texture bias.

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/OliverStoll/Texture-Bias-DL.git
   cd Texture-Bias-DL
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

To analyze texture bias in a model:

1. **Prepare your dataset**: Place your dataset in the `data/` directory and add the dataloader under `src/data_loading`.

2. **Add custom models**: Add further CNN or Transformer models in the `models/models.py` ModelFactory to include your model.

3. **Training of models**: Train all models via the `RunManager`
   
4. **Analyze texture bias**: Run the analysis scripts under `evaluation/extractor` to aggregate run results and plot them via `evaluation/plots` `PlotManager` to visualize feature bias.

## 📈 Results

The analysis tools will generate plots and metrics indicating the presence and extent of texture bias in your model. These results can be found in the `visualizations/` directory.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or feedback, please contact [Oliver Stoll](mailto:oliverstoll.berlin@gmail.com).
