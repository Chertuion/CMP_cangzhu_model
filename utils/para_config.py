import argparse


def init_args():
    parser = argparse.ArgumentParser('graph Mutual Information for OOD')
    
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='/data/home/wxl22/changzu_ans/datasets',type=str, help='root for datasets')
    parser.add_argument('--check_point', default='/data/home/wxl22/changzu_ans/checkpoints',type=str, help='check point for datasets')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    
    parser.add_argument('--component', default="pareto_value", type=str, help='component name, one of 32')
    parser.add_argument('--log_dir', default='/data/home/wxl22/changzu_ans/logs', type=str, help='log directory')
    parser.add_argument('--metric', default='rmse', type=str, help='metric')
    parser.add_argument('--model', choices=['1dcnn','lstm','gru', 'mambanir', 'hashmambanir'], default='1dcnn', type=str, help='model name')
    parser.add_argument('--save_model', default=True, type=bool, help='save model')
    
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=15, type=int, help='training iterations')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate for the predictor')
    parser.add_argument('--k_fold', default=2, type=int)

    parser.add_argument('--pareto', default=True, type=bool)
    parser.add_argument('--hotMap', default="no", type=str)
    parser.add_argument('--real_split', default="no", type=str)
    parser.add_argument('--best_r2', default=1, type=int)
    parser.add_argument('--goal', default= "pred", choices=["train", "vis", "pred"], help="choose the goal for model")

    parser.add_argument('--bins_method', default="jenks", type=str, choices=["quantile", "equal", "jenks", "kmeans", "zscore", "fixed"])
    parser.add_argument('--bins_n', default=5, type=int)
    parser.add_argument('--grid_res', default=0.01, type=float)
    parser.add_argument('--draw_mode', default="IDW", type=str, choices=["IDW", "Kriging"])
    parser.add_argument('--is_filt', default=False, type=bool)


    args = parser.parse_args()
    return args
