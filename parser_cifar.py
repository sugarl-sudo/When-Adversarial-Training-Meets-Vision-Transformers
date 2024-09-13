import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--method', type=str, default='pgd',
                        choices=['AT', 'TRADES', 'MART'])
    parser.add_argument('--dataset', type=str, default="cifar")
    parser.add_argument('--run-dummy', action='store_true')
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--AA-batch', default=128, type=int,help="Batch size for AA.")
    parser.add_argument('--crop', type=int, default=32)
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--n_w', type=int, default=10)
    parser.add_argument('--attack-iters', type=int, default=10, help='for pgd training')
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--ARD', action='store_true')
    parser.add_argument('--PRM', action='store_true')
    parser.add_argument('--drop-rate', default=1.0, type=float)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument('--eval-restarts', type=int, default=1)
    parser.add_argument('--eval-iters', type=int, default=10)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--labelsmoothvalue', default=0, type=float)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
                        help='Perturbation initialization method')
    parser.add_argument('--out-dir', '--dir', default='./log', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.3,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--body-freaze', action='store_true', help='Freeze the body of the model')
    parser.add_argument('--lbgat', action='store_true', help='Use LBGAT loss')
    parser.add_argument('--lbgat-beta', type=float, default=1.0)
    parser.add_argument('--teacher-model-path', type=str, default='')
    parser.add_argument('--mse-rate', type=float, default=1.0)
    parser.add_argument('--features', action='store_true', help='Use features for the model')
    args = parser.parse_known_args()[0]
    assert args.batch_size % args.accum_steps == 0
    return args