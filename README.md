# An Empirical Study on Fine-tuning Large Language Models of Code for Automated Program Repair

<p aligh="center">
This repository contains the code for <b> An Empirical Study on Fine-tuning Large Language Models of Code for Automated Program Repair </b> and the Page (https://LLMC4APR) that has some visualizd data.
</p>



# Experimental Checkpoint, Result, and Data: https://drive.google.com/drive/folders/1-3bA4fkvi18Pl9daIAhqUY4s7_tEph-d?usp=sharing


## Dependency
* Python 3.10.8
* PyTorch 1.13.1
* Huggingface transformers 4.24.0
* Tree-sitter 0.20.1
* Java 8
* Defects4J



## LLMC4APR
The file structure of the artifact is as follow:
### [Source Code](https://drive.google.com/drive/folders/1-3bA4fkvi18Pl9daIAhqUY4s7_tEph-d?usp=sharing)
* **Code:**
    * **CodeBERT:** source code for model fine-tuning and inference.
    * **GraphCodeBERT:** source code for model fine-tuning and inference.
    * **PLBART:** source code for model fine-tuning and inference.
    * **CodeT5:** source code for model fine-tuning and inference.
    * **UniXcoder:** source code for model fine-tuning and inference.
 ### [Experimental Data & Results](https://drive.google.com/drive/folders/1-3bA4fkvi18Pl9daIAhqUY4s7_tEph-d?usp=sharing)
 * **Dataset:**
    * **Tufano_dataset:** [BFP dataset](https://sites.google.com/view/learning-fixes/data), model checkpoints, candidate patches (BFP dataset).
    * **SequenceR_dataset:** [SequenceR dataset](https://github.com/ASSERT-KTH/sequencer/tree/master/data), model checkpoints, candidate patches (SequenceR dataset).
    * **Recoder_dataset:** [Recoder dataset](https://doi.org/10.5281/zenodo.7559208), model checkpoints, candidate patches (Defects4J V1.2 and V2.0).
    * **CPatMiner_dataset:** [CPatMiner dataset](https://drive.google.com/open?id=1M_0dRYqhCMh26GQbnX4Igp_2jSrTS1tV), model checkpoints, candidate patches (Defects4J V1.2).
    * **VulRepair_dataset:** [VulRepair dataset](https://github.com/awsm-research/VulRepair/tree/main/data/fine_tune_data), model checkpoints, candidate patches (VulRepair dataset).
    * **TFix_dataset:** [TFix dataset](https://drive.google.com/file/d/1CtfnYaVf-q6FZP5CUM4Wh7ofpp8b9ajW/view?usp=sharing), model checkpoints, candidate patches (TFix dataset).
    * **Defects4J_dataset:** [Defects4J dataset](https://github.com/rjust/defects4j).
    
    
    
    
## Reproduction
Download source code and datasets from https://drive.google.com/drive/folders/1-3bA4fkvi18Pl9daIAhqUY4s7_tEph-d?usp=sharing.
### Model Fine-tuning and Inference:
    
    cd Code
    
    # Select the model (CodeBERT/GraphCodeBERT/PLBART/CodeT5/UniXcoder) to be fine-tuned
    # Here is an example of CodeBERT
    cd CodeBERT
    
    # Run Task1 (BFP dataset)
    bash train_bfp.sh
    bash test_bfp.sh
    
    # Run Task2 (SequenceR dataset)
    bash train_sequencer.sh
    bash test_sequencer.sh
    
    # Run Task3 (Recoder dataset)
    bash train_recoder.sh
    bash test_recoder.sh
    
    # Run Task4 (CPatMiner dataset)
    bash train_cpm.sh
    bash test_cpm.sh
    
    # Run Task5 (VulRepair dataset)
    bash train_vul.sh
    bash test_vul.sh
    
    # Run Task6 (TFix dataset)
    bash train_tfix.sh
    bash test_tfix.sh
    








    
