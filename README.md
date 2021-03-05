# GAIRAT-LSA
Evaluating GAIRAT robustness using Logit Scaling Attack.
We evaluate the adversarial robustness of a very recent method called ["Geometry-aware Instance-reweighted Adversarial Training"](https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training) using Logit Scaling Attack [1,2,3].

### Overview
This very simple attack has been firstly shown by Carlini and Wagner to break robustness by distillation in [1] and was, later on, employed in [2]. Though this attack has been posted since 2016, it is not well known in the community.
In our tech report, we show that method such as GAIRAT, that scales the loss during training, are subject to the gradient masking that can be revealed with this type of attack.

The results of our experiments can be found [here](https://arxiv.org/abs/2103.01914).

To test GAIRAT on CIFAR-10 we had to train their model and our pre-trained models can be found [here](https://drive.google.com/drive/folders/1vSPEmYtilhsj3jFJk25VVTQEouGLpWnV?usp=sharing).

### Usage

[Download](https://drive.google.com/drive/folders/1vSPEmYtilhsj3jFJk25VVTQEouGLpWnV?usp=sharing) our pre-trained model. 

Then:

    pip install tqdm torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

to install the needed dependencies. We tested using PyTorch 1.7.1 and CUDA 11.0.


    python eval_pgd.py --model_path <model_path> --output_suffix=<result_path> --num_restarts 1 --num_steps 20 --alpha <alpha>

to test the model using different values of `alpha` (the scaling factor). 
* `<model_path>` is the pre-trained model path (e.g. checkpoint.pth.tar).
* `<result_path>` is the path where to store evaluation results.
* `<alpha>` is the desired scaling value (e.g. 10.0).

# Citations

If you find our work useful, please cite:

```latex
@article{zhang2020geometry,
  title={Evaluating the Robustness of Geometry-Aware Instance-Reweighted Adversarial Training},
  author={Hitaj,Dorjan and Pagnotta, Giulio and Masi, Iacopo and Mancini, Luigi V.}
  journal={arXiv preprint arXiv:2103.01914},
  year={2021}
}

```

### Reference
[1] [https://arxiv.org/pdf/1607.04311.pdf](https://arxiv.org/pdf/1607.04311.pdf)

[2] https://arxiv.org/pdf/2003.01690.pdf

[3] https://arxiv.org/pdf/2103.01914.pdf
