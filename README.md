Title: Evolving Graph Structure for Attribute-missing Semi-supervised Node Classification

Brief Overview: This paper proposes the Evolutionary Graph Structure (EGS) for semi-supervised node classification with attribute missing, addressing the issue of being unable to effectively and dynamically describe and update the collected structural relationships during the process of reconstructing missing node attributes from both attribute and structural perspectives.

Running the Code:
1. Prepare the datasets The dataset can be automatically downloaded to run in src/run. If there are any issues during the process, you can also directly download it through the link https://github.com/kimiyoung/planetoid.
2. Environmental requirements. 
python==3.8 
cuda==11.3 
torch==1.10.0 
pytorch_geomtric==2.0.4 
ogb==1.3.5
3. Quick start.
   (1) Path Configuration: Set the working directory to the project root based on your experimental setup.
   (2) Run Main Program: Execute the following command to initiate model training and evaluation: python src/run.py --model [model_name] --filling_method [filling_method] --dataset_name [dataset] --missing_rate [rate] --lambda_1 [value] --lambda_2 [value]
   (3) Parameter Description:
    (a)model_name: Graph neural network model
    (b)filling_method: Missing value imputation method
    (c)dataset: Dataset name
    (d)missing_rate: Missing data ratio
    (e)lambda_1, lambda_2: Regularization hyperparameters.
Example: python src/run.py --model gcn --filling_method featureanddeta_propagation --dataset_name CiteSeer --missing_rate 0.05 --lambda_1 0.1 --lambda_2 0.1
