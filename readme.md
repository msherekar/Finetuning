* This repository is for fine-tuning ESM-2 & ESM-3 models *
The goal of finetuning is for customizing the model for a desired task 
The desired task could be prediction of solubility, binding, safety etc.
Finetuning could be done in one of two ways-
1.) embeddings
2.) masked language modeling via a masking function
Practically speaking, the head layer of the model is customized.
In this current set-up, inputs will be three .csv files (training, validation, testing)
And output will be finetuned model, metrics, plots