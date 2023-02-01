import argparse
def parse_args():
    parser=argparse.ArgumentParser()
    ddp = parser.add_argument_group('ddp')
    ddp.add_argument('--distributed',default=1,type=int)
    ddp.add_argument("--local_rank", type=int,                                    
                            help='rank in current node')                                 
    ddp.add_argument('--use_mix_precision', default=False,                        
                            action='store_true', help="whether to use mix precision") 
    train = parser.add_argument_group('train')
    train.add_argument('--gpu',default=0,type=int)
    train.add_argument('--batch_size',default=4,type=int)
    train.add_argument('--n_epoch',default=5,type=int)
    train.add_argument('--lr',default=1e-4,type=float)
    train.add_argument('--eval_step',default=20000,type=int)
    train.add_argument('--seed',default=42,type=int)

    trainer = parser.add_argument_group('trainer')
    trainer.add_argument('--eval_strategy',default='epoch',type=str)
    trainer.add_argument('--output_dir',default='/share/zhangzk/Organized_Coding/demo/eval',type=str)
    trainer.add_argument('--mode',default="normal",type=str)

    args=parser.parse_args()
    return args

def getname(args):
    name = f'Model_{args.seed}_{args.lr}'
    return name