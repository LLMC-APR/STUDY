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
### Source Code
* **Code:**
    * **CodeBERT:** source code for model fine-tuning and inference.
    * **GraphCodeBERT:** source code for model fine-tuning and inference.
    * **PLBART:** source code for model fine-tuning and inference.
    * **CodeT5:** source code for model fine-tuning and inference.
    * **UniXcoder:** source code for model fine-tuning and inference.
 ### [Experimental Data & Results](https://drive.google.com/drive/folders/1-3bA4fkvi18Pl9daIAhqUY4s7_tEph-d?usp=sharing)
 * **Dataset:**
    * **CPatMiner_dataset:** [CPatMiner dataset](https://drive.google.com/open?id=1M_0dRYqhCMh26GQbnX4Igp_2jSrTS1tV), model checkpoints, candidate patches (Defects4J V1.2).
    * **Recoder_dataset:** [Recoder dataset](https://doi.org/10.5281/zenodo.7559208), model checkpoints, candidate patches (Defects4J V1.2 and V2.0).
    * **SequenceR_dataset:** [SequenceR dataset](https://github.com/ASSERT-KTH/sequencer/tree/master/data), model checkpoints, candidate patches (SequenceR dataset).
    * **TFix_dataset:** [TFix dataset](https://drive.google.com/file/d/1CtfnYaVf-q6FZP5CUM4Wh7ofpp8b9ajW/view?usp=sharing), model checkpoints, candidate patches (TFix dataset).
    * **Tufano_dataset:** [BFP dataset](https://sites.google.com/view/learning-fixes/data), model checkpoints, candidate patches (BFP dataset).
    * **VulRepair_dataset:** [VulRepair dataset](https://github.com/awsm-research/VulRepair/tree/main/data/fine_tune_data), model checkpoints, candidate patches (VulRepair dataset).
    
