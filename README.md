
# Solar Panel Segmentation

## Overview
This project utilizes semantic segmentation techniques to identify solar panels in satellite images. By leveraging deep learning models, this project aims to accurately segment solar panels to facilitate analysis on solar energy usage and infrastructure.

## Repository Structure
- **data/**: Contains datasets used in training and evaluating the model.
- **notebooks/**: Jupyter notebooks with exploratory data analysis and model training experiments.
- **src/**: Source code for model training, evaluation, and utility functions.
- **models/**: Contains trained models ready for deployment or further training.
- **results/**: Storage for model outputs and logs.
- **main.py**: Main script to run the training pipeline.
- **train.py**: Training procedures for the segmentation model.
- **test.py**: Testing procedures to evaluate the model.

## Getting Started
### Prerequisites
- Python 3.12
- pipenv (for dependency management)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tkescec-algebra/solar_panel_segmentation.git
   cd solar_panel_segmentation
   ```
2. Install dependencies:
   ```bash
   pipenv install
   ```

### Running the Code
To train the model, run:
```bash
python train.py
```
For testing the model on new images, run:
```bash
python test.py
```

## Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Authors
- Tomislav Keščec

## Acknowledgments
Thanks to everyone who has contributed to the development and testing of this project!
