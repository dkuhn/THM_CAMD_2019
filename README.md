# THM_CAMD_2019
Materials for the Computer-aided molecular design course at THM Gießen WS2019/2020

In this practical exercise between both lecturing blocks you will become a computational chemist. You will collect affinity data for a protein kinase target. If you do not know what protein kinase target to work on, please drop me an email. 

Please perform the follwing steps:

1. Go to Uniprot find out human Uniprot identifier for your protein kinase.
2. Select IC50 assays for target in Chembl and remove PAINS compounds
3. Create a pandas dataframe containing IC50 data and keep values with operators
4. Average IC50 data for values without operator. In case of IC50 values with operators at different ligand concentrations use just one, the one with more information content: >10, >1 --> choose >10. In case you have both IC50 values for the same compound with and without operator just consider IC50 values without operator.
5. Create training and test dataset (20%)
6. Build five different categorical ML models predicting kinase activity using different scikit-learn learners. Use 1uM as activity threshold
7. Analyse models using accuracy, sensitivity and specificity using cross-validation. Check your final model best model on the test data set.
8. Select one model with good recall and one model with good precision
9. Create another training/test split and build a regression model. Select best model based on AUC.


Materials used
For each step please re-visit the excellent [TeachOpenCADD talktorials](https://github.com/volkamerlab/TeachOpenCADD "TeachOpenCADD") jupyter notebooks that we have discussed during our lecture.

You can work with jupyter notebooks, finally please provide a python script (can be based on the jupyter notebooks) that can be executed from command-line and does every analysis step. Make PAINS filtering optional

Thanks and enjoy!
