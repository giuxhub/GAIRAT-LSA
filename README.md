# GAIRAT-LSA
Evaluating GAIRAT robustness using Logit Scaling Attack.
We evaluate the adversarial robustness of a very recent method called ["Geometry-aware Instance-reweighted Adversarial Training"](https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training) using Logit Scaling Attack. 

The results of our experiments can be found [here](https://arxiv.org/abs/2103.01914).

To test GAIRAT on CIFAR-10 we had to train their model and our pre-trained models can be found [here](https://drive.google.com/drive/folders/1vSPEmYtilhsj3jFJk25VVTQEouGLpWnV?usp=sharing).

### Usage #1

[Download](https://drive.google.com/drive/folders/1vSPEmYtilhsj3jFJk25VVTQEouGLpWnV?usp=sharing) our pre-trained model. 

Then:

    pip install tqdm torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

to install the needed dependencies. We tested using PyTorch 1.7.1 and CUDA 11.0.


    python eval_pgd.py --model_path <model_path> --output_suffix=<result_path> --num_restarts 1 --num_steps 20 --alpha <alpha>

to test the model using different alphas. <model_path> is the pre-trained model path (e.g. checkpoint.pth.tar) and <alpha> is the desired scaling value (e.g. 10.0).

### Usage #2

Import `gairat-lsa.ipynb` in Google Colab and run it.