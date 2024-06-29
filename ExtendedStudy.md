
# Extended Study

<p aligh="center">
This repository contains the code and data for the Extended Study.
   
Experimental Code and Data: https://drive.google.com/drive/folders/12xIiZChs81NpUjGRCxHbuQEuT2ozHNB9?usp=sharing
</p>

## Dependency
* Python 3.10.8
* PyTorch 1.13.1
* Transformers
* Peft
* Bitsandbytes
* Accelerate
* Sentencepiece


## LLM4APR study (Task 7)
The file structure of the artifact is as follow:
### [Source Code]
* **Code**
    * **1_LLM4APR**
        * **CodeGeeX2:** source code for model fine-tuning and inference.
        * **CodeGen25:** source code for model fine-tuning and inference.
        * **CodeLlama13B:** source code for model fine-tuning and inference.
        * **CodeLlama70B:** source code for model fine-tuning and inference.
        * **InCoder:** source code for model fine-tuning and inference.
        * **StarCoder:** source code for model fine-tuning and inference.
        * **StarCoder2:** source code for model fine-tuning and inference.
 ### [Experimental Data & Results]
 * **Data:**
    * **Transfer_dataset:** [Transfer dataset](https://dl.acm.org/doi/10.1145/3510003.3510147).
        * **2-Mark2:** training dataset.
        * **Result:**
            * **1_LLM4APR:** find-tuned LLMs and repair result.
    * **Defects4J_dataset:** [Defects4J V1.2 dataset](https://github.com/rjust/defects4j).
    * **HumanEval_dataset:** [HumanEval-Java dataset](https://github.com/lin-tan/clm/tree/main/humaneval-java).
        
### Reproduction
Download source code and datasets from https://drive.google.com/drive/folders/12xIiZChs81NpUjGRCxHbuQEuT2ozHNB9?usp=sharing.
<b> Model Fine-tuning and Inference: </b>
    
    cd Code/1_LLM4APR
    
    # Select the model (InCoder/CodeGeeX2/CodeGen25/CodeLlama13B/CodeLlama70B/StarCoder/StarCoder2) to be fine-tuned
    # Here is an example of StarCoder
    cd StarCoder/MARK2
    
    # Run Task7 
    bash train.sh     # Training on Transfer dataset
    bash merge.sh     # Merge Adapter
    bash test_d4j.sh  # Testing on Defects4J V1.2
    bash test_hev.sh  # Testing on HumanEval-Java



## PEFT4LLM study
The file structure of the artifact is as follow:
### [Source Code]
* **Code**
    * **2_PEFT4LLM**
        * **CodeGeeX2:** source code for model fine-tuning and inference with 3 PEFT techniques and the FPFT strategy.
        * **CodeGen25:** source code for model fine-tuning and inference with 3 PEFT techniques and the FPFT strategy.
        * **CodeLlama:** source code for model fine-tuning and inference with 3 PEFT techniques and the FPFT strategy.
        * **InCoder:** source code for model fine-tuning and inference with 3 PEFT techniques and the FPFT strategy.
        * **StarCoder:** source code for model fine-tuning and inference with 3 PEFT techniques and the FPFT strategy.
 ### [Experimental Data & Results]
 * **Data:**
    * **Transfer_dataset:** [Transfer dataset](https://dl.acm.org/doi/10.1145/3510003.3510147).
        * **1-Mark2:** training dataset.
        * **Result:**
            * **2_PEFT4LLM:** find-tuned LLMs and repair result.

        
### Reproduction
Download source code and datasets from https://drive.google.com/drive/folders/12xIiZChs81NpUjGRCxHbuQEuT2ozHNB9?usp=sharing.
<b> Model Fine-tuning and Inference: </b>
    
    cd Code/2_PEFT4LLM
    
    # Select the model (InCoder/CodeGeeX2/CodeGen25/CodeLlama/StarCoder) to be fine-tuned
    # Here is an example of StarCoder
    cd StarCoder
    
    # Fine-tuning with ADALoRA techniques
    cd ADALORA
    bash train.sh     
    bash merge.sh     
    bash test.sh

    # Fine-tuning with IA3 techniques
    cd IA3
    bash train.sh     
    bash merge.sh     
    bash test.sh

    # Fine-tuning with LORA techniques
    cd LORA
    bash train.sh     
    bash merge.sh     
    bash test.sh

    # Fine-tuning with FPFT strategy
    cd FULL
    bash train_and_test.sh     


## APR4LLM study
The file structure of the artifact is as follow:
### [Source Code]
* **Code**
    * **3_APR4LLM**
        * **CodeGeeX2:** source code for model fine-tuning and inference with 4 repair stategies.
        * **CodeGen25:** source code for model fine-tuning and inference with 4 repair stategies.
        * **CodeLlama:** source code for model fine-tuning and inference with 4 repair stategies.
        * **InCoder:** source code for model fine-tuning and inference with 4 repair stategies.
        * **StarCoder:** source code for model fine-tuning and inference with 4 repair stategies.
 ### [Experimental Data & Results]
 * **Data:**
    * **Transfer_dataset:** [Transfer dataset](https://dl.acm.org/doi/10.1145/3510003.3510147).
        * **2-Mark2:** training dataset.
        * **2-ITER:** training dataset.
        * **2-TENURE:** training dataset.
        * **2-KATANA:** training dataset.
        * **Result:**
            * **3_APR4LLM:** find-tuned LLMs and repair result.

        
### Reproduction
Download source code and datasets from https://drive.google.com/drive/folders/12xIiZChs81NpUjGRCxHbuQEuT2ozHNB9?usp=sharing.
<b> Model Fine-tuning and Inference: </b>
    
    cd Code/3_APR4LLM
    
    # Select the model (InCoder/CodeGeeX2/CodeGen25/CodeLlama/StarCoder) to be fine-tuned
    # Here is an example of StarCoder
    cd StarCoder
    
    # Fine-tuning with the ITER
    cd ITER
    bash train.sh     
    bash merge.sh     
    bash test_d4j_iter.sh

    # Fine-tuning with the KATANA
    cd KATANA
    bash train.sh     
    bash merge.sh     
    bash test_d4j.sh

    # Fine-tuning with the basic NMT
    cd MARK3
    bash train.sh     
    bash merge.sh     
    bash test_d4j.sh

    # Fine-tuning with the TENURE
    cd TENURE
    bash train.sh
    bash merge.sh
    bash test_d4j.sh   
    

    

