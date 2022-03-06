# RanMASK

The code is for Paper: Certified Robustness to Text Adversarial Attacks by Randomized [MASK] https://arxiv.org/pdf/2105.03743.pdf

This code is based on [textattack](https://github.com/QData/TextAttack) and we make some changes, e.g., add a HuggingFaceModelMaskEnsembleWrapper for situations where ensemble methods is required.


This code supports 3 base mode, train, evalute and attack. For RanMASK, it supports certify. (Seeing the args.py)


<br /><br />
## train
You can train a base classifer using

    python main.py --mode train --dataset_name sst2

For RanMASK, you can use

    python main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3

where the mask rate is 0.3.
<br /><br />
## evaluate

For RanMASK, you can use

    python main.py --mode evaluate --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3

to evaluate a model.

Full texts are used (not masked) by default under the evalute mode. However, in RanMASK, masked texts are used while training. Thus there is a inconsistency problem between training and prediction.


<br /><br />
## attack
You can attack a trained classifer using  (the attacker is implemented via textattack)
You can train a base classifer using

    python main.py --mode attack --dataset_name sst2 --attack_method pwws --attack_numbers 1000

For RanMASK, you can use

    python main.py --mode attack --dataset_name sst2 --attack_method textfooler --training_type sparse --sparse_mask_rate 0.3 --attack_numbers 1000

where the attacker is chosen by attack_method parameter.

<br /><br />
## certify
For RanMASK, you can use

    python main.py --mode certify --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3 --certify_numbers 1000


For more details of other parameters, please refer to args.py.