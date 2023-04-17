# Capstone_GMT89
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
pip install -r requirements.txt

## usage
Version 1.0.0
To run the model:

0. Download the [latest version](https://github.com/foye501/Capstone_GM89/releases)
1. Go to `Project_Airbnb2/Scripts/the_pipelines`
2. `python update_features.py`. This will generate processing pipelines and features that will be utilized in the training.
3. `python train.py`. This will train all five quantile regression LightGBM model, and store them. Other preprocessing like imputation, dimensionality reduction ate also involved. 
4. Go to `../Capstone_app`, run `streamline run Home.py` to launch the web application. You can directly go to our web platform: http://18.205.39.151:8502


### Overall dataflow
![Overall_workflow](./Project_Airbnb2/assets/images/Overall_workflow.png)

### Amenity Pipeline
![Amenitypipeline](./Project_Airbnb2/assets/images/Amenity.png)

### Image Pipeline
![Imagepipeline](./Project_Airbnb2/assets/images/imagepipeline.png)

### location_pipeline
![locationpipeline](./Project_Airbnb2/assets/images/location_pipeline.png)

### NLP_pipeline
![NLP_pipeline](./Project_Airbnb2/assets/images/NLP_pipeline.png)

### Contributing

- Wei Li Tan: Amenities analysis, final report consolidation
- Yangkang Chen: Image analysis, model consolidation
- Masato Ando: Location analysis, github
- Dongyao Wang: Description NLP analysis, Web application


