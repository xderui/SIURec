import argparse

def parse_args(args=[]):
    parser = argparse.ArgumentParser(description="SIURec")
    # dataset
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--dataset', default='ml-1m')

    # model
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')

    # train
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--n_batch', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--emb_reg', type=float, default=2.5e-5)
    parser.add_argument('--uniform_reg', type=float, default=0.1)

    # eval
    parser.add_argument('--topk', nargs='?', default='[20, 40]')

    return parser.parse_args()