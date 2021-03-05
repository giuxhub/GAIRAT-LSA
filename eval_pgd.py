"""
Evaluate robustness against specific attack.
Based on code from https://github.com/yaircarmon/semisup-adv
"""

import os
import numpy as np
import argparse
import logging
import torch
from torch.autograd import Variable
from datasets import SemiSupervisedDataset
from torchvision import transforms
from attack_pgd import pgd
import torch.backends.cudnn as cudnn
from utils_eval import get_model
from tqdm import tqdm


def eval_adv_test(model, device, test_loader, attack, attack_params,
                  results_dir, num_eval_batches, alpha):
    """
    evaluate model by white-box attack
    """
    model.eval()
    if attack == 'pgd':
        restarts_matrices = []
        for restart in range(attack_params['num_restarts']):
            is_correct_adv_rows = []
            count = 0
            batch_num = 0
            natural_num_correct = 0
            for data, target in tqdm(test_loader):
                batch_num = batch_num + 1
                if num_eval_batches and batch_num > num_eval_batches:
                    break
                data, target = data.to(device), target.to(device)
                count += len(target)
                X, y = Variable(data, requires_grad=True), Variable(target)
                # is_correct_adv has batch_size*num_iterations dimensions
                is_correct_natural, is_correct_adv = pgd(
                    model, X, y,
                    epsilon=attack_params['epsilon'],
                    num_steps=attack_params['num_steps'],
                    step_size=attack_params['step_size'],
                    random_start=attack_params['random_start'], alpha=alpha)
                natural_num_correct += is_correct_natural.sum()
                is_correct_adv_rows.append(is_correct_adv)

            is_correct_adv_matrix = np.concatenate(is_correct_adv_rows, axis=0)
            restarts_matrices.append(is_correct_adv_matrix)

            is_correct_adv_over_restarts = np.stack(restarts_matrices, axis=-1)
            num_correct_adv = is_correct_adv_over_restarts.prod(
                axis=-1).prod(axis=-1).sum()

            logging.info("Accuracy after %d restarts: %.4g%%" %
                         (restart + 1, 100 * num_correct_adv / count))
            stats = {'attack': 'pgd',
                     'count': count,
                     'attack_params': attack_params,
                     'natural_accuracy': natural_num_correct / count,
                     'is_correct_adv_array': is_correct_adv_over_restarts,
                     'robust_accuracy': num_correct_adv / count,
                     'restart_num': restart
                     }

            np.save(os.path.join(results_dir, 'pgd_results.npy'), stats)

    else:
        raise ValueError('Unknown attack %s' % attack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices='cifar10',
                        help='The dataset')
    parser.add_argument('--model_path',
                        help='Model for attack evaluation')
    parser.add_argument('--model', '-m', default='wrn-32-10', type=str,
                        help='Name of the model: wrn-XX-XX, resnet-XX, small-cnn')
    parser.add_argument('--output_suffix', default='', type=str,
                        help='String to add to log filename')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='Input batch size for testing (default: 200)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='Attack perturbation magnitude')
    parser.add_argument('--attack', default='pgd', type=str,
                        help='Attack type',
                        choices=('pgd'))
    parser.add_argument('--num_steps', default=40, type=int,
                        help='Number of PGD steps')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='PGD scale logit factor')
    parser.add_argument('--step_size', default=0.01, type=float,
                        help='PGD step size')
    parser.add_argument('--num_restarts', default=5, type=int,
                        help='Number of restarts for PGD attack')
    parser.add_argument('--no_random_start', dest='random_start',
                        action='store_false',
                        help='Disable random PGD initialization')
    parser.add_argument('--random_seed', default=0, type=int,
                        help='Random seed for permutation of test instances')
    parser.add_argument('--num_eval_batches', default=None, type=int,
                        help='Number of batches to run evalaution on')
    parser.add_argument('--shuffle_testset', action='store_true', default=False,
                        help='Shuffles the test set')

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)

    output_dir, checkpoint_name = os.path.split(args.model_path)

    results_dir = os.path.join('./', args.output_suffix)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    epoch = 0
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(results_dir,
                                             'attack_epoch%d%s.log' %
                                             (epoch, args.output_suffix))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    logging.info('Attack evaluation')
    logging.info('Args: %s' % args)

    # settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dl_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # set up data loader
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                    train=False, root='./data/cifar-10',
                                    download=True,
                                    transform=transform_test)

    if args.shuffle_testset:
        np.random.seed(123)
        logging.info("Permuting testset")
        permutation = np.random.permutation(len(testset))
        testset.data = testset.data[permutation, :]
        testset.targets = [testset.targets[i] for i in permutation]

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.batch_size,
                                              shuffle=False, **dl_kwargs)

    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=normalize_input)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if not all([k.startswith('module') for k in state_dict]):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
    else:
        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s


        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    attack_params = {
        'epsilon': args.epsilon,
        'seed': args.random_seed
    }
    if args.attack == 'pgd':
        attack_params.update({
            'num_restarts': args.num_restarts,
            'step_size': args.step_size,
            'num_steps': args.num_steps,
            'random_start': args.random_start,
        })

    logging.info('Running %s' % attack_params)
    eval_adv_test(model, device, test_loader, attack=args.attack,
                  attack_params=attack_params, results_dir=results_dir,
                  num_eval_batches=args.num_eval_batches, alpha=args.alpha)
