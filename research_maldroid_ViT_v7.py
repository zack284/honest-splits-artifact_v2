import argparse
import logging
import glob
import os
import random
import torch
from torch import nn
from torch.optim import SGD
from tensorboardX import SummaryWriter
#from dataset.semi_BUSI import load_dataloader, get_subset_dataloaders
from dataset.semi_Maldroid import load_dataloader, get_subset_dataloaders
from model.semseg.deeplabv3plus import branch_container, DeepLabV3Plus_Classification
from util.utils import init_log_save, EMA,  one_epoch, one_epoch_co, save_best_model, evaluate, evaluate_v2


import math
from torch.optim.lr_scheduler import LambdaLR





######
######     python research_maldroid_Fixmatch_gradCAM.py --origin --use_con --save_path=./results/temp/temp_v2
######

parser = argparse.ArgumentParser(description='SSL')
# default settings
parser.add_argument('--root', default='D:\code\BUSI', type=str)

parser.add_argument('--backbone', default='ViT', type=str)


parser.add_argument('--labeled_id_path',   default='./splits_ssl/train_labeled.txt', type=str)
parser.add_argument('--unlabeled_id_path', default='./splits_ssl/train_unlabeled.txt', type=str)

parser.add_argument('--val_id_path', default='./splits_ssl/val_combined.txt', type=str)
# --- 추가 ---
parser.add_argument('--test_clean_id_path', default='./splits_ssl/test_clean.txt', type=str)
parser.add_argument('--test_obf_id_path',   default='./splits_ssl/test_obf.txt', type=str)
parser.add_argument('--manifest', default='~/image_pairing_manifest.csv', type=str)
parser.add_argument('--p_obf', default=0.0, type=float,help='원본을 실제 난독화본으로 교체할 확률. 0=baseline')


parser.add_argument('--save_path', default='results', type=str)
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--nclass', default=5, type=int)
parser.add_argument('--threshold', default=0.95, type=float)
#parser.add_argument('--threshold', default=0.25, type=float)
parser.add_argument('--batch_size', default=6, type=int)
parser.add_argument('--seed', default=4, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--origin', action='store_true')
parser.add_argument('--free',action='store_true') # freematch 실험 인자
# logic settings
parser.add_argument('--co_train', action='store_true') # it affects to learning process 
parser.add_argument('--ring', action='store_true') # it affects to learning process
# network settings
parser.add_argument('--mode_mapping', default='both', type=str)
parser.add_argument('--use_MLP', action='store_true')
parser.add_argument('--ema', action='store_false')
# loss settings
parser.add_argument('--use_con', action='store_true') 
parser.add_argument('--use_osv', action='store_true') # this parameter require that co_train is true
# weight settings
parser.add_argument('--w_CE', default=1.0, type=float)
parser.add_argument('--w_con', default=0.9, type=float)
# optimizer settings
parser.add_argument('--base_lr', default=0.0001, type=float)                          
parser.add_argument('--mul_scheduler', default=0.9, type=float)
parser.add_argument('--restore_lr', action='store_true') # this parameter for ring parameter



args = parser.parse_args()


# 2. Define Cosine Schedule with Warmup
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        # Linear Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine Decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main(gpu, args):
    args.local_rank = gpu
    print("I am in ",gpu)
    print(f'Epochs:{args.epochs} '\
        f'EMA:{args.ema} '\
        f'Weak-Strong consistency:{args.use_con} '\
        f'Othersideview:{args.use_osv} '\
        f'Co-train:{args.co_train} '\
        f'Ring subset:{args.ring} '\
        f'MLP:{args.use_MLP} '\
        f'Restore learning rate:{args.restore_lr} ')
    if args.local_rank <= 0:
        os.makedirs(args.save_path, exist_ok=True)
        
    logger = init_log_save(args.save_path, 'global', logging.INFO)
    logger.propagate = 0

    if args.local_rank <= 0:
        tb_dir = args.save_path
        tb = SummaryWriter(log_dir=tb_dir)

    # setting seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.co_train:
        model = branch_container(args)
    else:
        if args.origin == False:
            model = DeepLabV3Plus_Classification(args, args.use_MLP)
        else:
            from torchvision import models
            from torchvision.models import Wide_ResNet101_2_Weights
            model = models.wide_resnet101_2(weights=Wide_ResNet101_2_Weights.DEFAULT)
            model.fc = nn.Linear(in_features=2048, out_features=args.nclass, bias=True)


# Assuming 'model' is your branch_container
    if args.co_train:
        print(f"Number of prefix tokens: {model.branch1.backbone.model.num_prefix_tokens}")

    logger.info('epoch: {}'.format(args.epochs))
    logger.info('ema: {}'.format(args.ema))
    logger.info('con: {}'.format(args.use_con))
    logger.info('osv: {}'.format(args.use_osv))
    logger.info('co-train: {}'.format(args.co_train))
    logger.info('ring: {}'.format(args.ring))
    logger.info('mlp: {}'.format(args.use_MLP))
    logger.info('restore lr: {}'.format(args.restore_lr))
    

    
    if args.ema == True:
        ema = EMA(model, 0.999)
    
    if args.co_train:


        optimizers = {'a':torch.optim.AdamW(model.branch1.parameters(), lr=1e-4, weight_decay=0.05,eps=1e-8),\
                      'b':torch.optim.AdamW(model.branch2.parameters(), lr=1e-4, weight_decay=0.05,eps=1e-8)}

    else:

        optimizers = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05,eps=1e-8) 



    print("logcal_rank",args.local_rank)
    if torch.cuda.is_available():
        model = model.cuda(args.local_rank)
    # ---- 
    # loss 
    # ---- 
    # CE loss for labeled data
    criterion_l = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1).cuda(args.local_rank)
    criterion_u = nn.CrossEntropyLoss(reduction='none',label_smoothing=0.1).cuda(args.local_rank)
    # ------- 
    # dataset 
    # ------- 
    trainloader_u, length_u = load_dataloader(args,'train_u')
    valloader =  load_dataloader(args,'val')

    valloader = load_dataloader(args, 'val')
    # --- 추가 ---
    testloader_clean = load_dataloader(args, 'test_clean')
    testloader_obf   = load_dataloader(args, 'test_obf')
    best_test_clean = 0.0
    best_test_obf   = 0.0
    best_val_epoch  = -1
    logger.info(f'p_obf = {args.p_obf}')
    

    previous_best = 0.0
    previous_best1 = 0.0
    previous_best2 = 0.0
    eval_mode = 'original'

    if args.ring==False: 
        trainloader_l = load_dataloader(args,'train_l', aug_num=length_u)
        print(f"Number of samples in labeled dataset: {len(trainloader_l.dataset)}")
    else: 
        file_pattern = os.path.join(args.labeled_id_path, "BUSI_training_labeled_subset_*.txt")
        file_list = sorted(glob.glob(file_pattern))
        random.shuffle(file_list)

        



        ASSIGNED_ORDER = None 
        
        if ASSIGNED_ORDER is not None:
            # Reorder file_list based on your assigned indices
            file_list = [file_list[i] for i in ASSIGNED_ORDER]
            if args.local_rank <= 0:
                logger.info(f"!!! MANUAL CURRICULUM ACTIVATED !!!")
        else:
            # Default: Shuffle randomly as before
            random.shuffle(file_list)
            logger.info(f"!!! RANDOM CURRICULUM ACTIVATED !!!")
            print('file_list',file_list)


# 2. Log the Final Order so you can "save" it for later
        if args.local_rank <= 0:
            order_names = [os.path.basename(f) for f in file_list]
            logger.info(f"--- [RING SUBSET ORDER] ---")
            logger.info(f"Order: {order_names}")
            # Also log the indices specifically to make it easy to copy into ASSIGNED_ORDER
            # We assume 'sorted(glob.glob)' gave us a base 0,1,2... order
            base_list = sorted(glob.glob(file_pattern))
            indices = [base_list.index(f) for f in file_list]

            
            logger.info(f"Indices for ASSIGNED_ORDER: {indices}")
        # -----------------------------
        
        # 각 서브셋이 사용될 epoch 크기 계산
        #sliding_window_size = int(0.4*args.epochs / len(file_list))
        sliding_window_size=20
        if not file_list:
            raise ValueError(f"No labeled_subset files found in {args.labeled_id_path}")
        if args.restore_lr:
            # cosine annealing을 각 subset에 적용하기 위함
            # 하나의 서브셋이 sws 만큼의 epoch를 거치고 또 unlabeled trainloader만큼 iteration 하니까 분모를 아래와 같이 계산
            denominator = int(sliding_window_size*len(trainloader_u)) 

    numerator = 0
    args.total_iters = len(trainloader_u) * args.epochs




# --- NEW SCHEDULER BLOCK START ---
    # Define warmup as 5% of total iterations or a fixed number of epochs (e.g., 20)
    # 20 epochs * iterations per epoch is a safe bet for DeiT on medical data

    # 주의: 아래 20 은 one_epoch 의 `epoch < 20` 과 반드시 같아야 한다.
    # 어긋나면 get_cosine_schedule_with_warmup 의 cosine 분기가 실행되는데
    # num_training_steps == num_warmup_steps 라 lr 이 매 step 1.0/0.0 으로 진동한다.


    num_warmup_steps = 20 * len(trainloader_u) 
    
    if args.co_train:
        # Wrap both branch optimizers in their own scheduler
        scheduler_a = get_cosine_schedule_with_warmup(
            optimizers['a'], 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_warmup_steps  # <--- Change this
        )
        scheduler_b = get_cosine_schedule_with_warmup(
            optimizers['b'], 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_warmup_steps  # <--- Change this
        )
        schedulers = {'a': scheduler_a, 'b': scheduler_b}
    else:
        # Single branch mode
        scheduler_single = get_cosine_schedule_with_warmup(
            optimizers, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_warmup_steps  # <--- Change this
        )
        schedulers = {'a': scheduler_single}    



    if args.free and args.origin:
        num_classes = args.nclass
        if not hasattr(args, 'p_t'):
            args.p_t = torch.ones(num_classes, device=args.local_rank) / num_classes
        if not hasattr(args, 'tau_t'):
            # init scalar tau as 1/C (float tensor on device)
            args.tau_t = torch.tensor(1.0 / num_classes, device=args.local_rank, dtype=args.p_t.dtype)
        if not hasattr(args, 'hist'):
            args.hist = torch.ones(num_classes, device=args.local_rank) / num_classes
        #if not hasattr(args, 'ema_decay'):
        #    args.ema_decay = 0.999
        args.ema_decay = getattr(args, 'ema_decay', 0.999)



    for epoch in range(args.epochs):


        if args.co_train==False:
            if args.ring and epoch%sliding_window_size==0: # 1 ~ n(length of file_list)
                trainloader_l = get_subset_dataloaders(args, file_list, epoch, sliding_window_size, length_u)

                if args.restore_lr:
                    print(f'numerator reach : {numerator} and denominator is : {denominator}')
                    numerator=0
                    optimizers.param_groups[0]['lr'] = args.base_lr

            mw_duration = 5
            mw_factor=1
            if args.ring and args.restore_lr:
                epoch_in_cycle = epoch % sliding_window_size

            if args.ring and args.restore_lr and epoch >= sliding_window_size:
            
                if epoch_in_cycle < mw_duration:
                    # Scale from 10% to 100%
                    mw_factor = 0.3 + (0.7 * (epoch_in_cycle / mw_duration-1))
                    optimizers.param_groups[0]['lr'] = args.base_lr * mw_factor
                    print(f"--- Micro-Warmup Active: Cycle Epoch {epoch_in_cycle}, Factor: {mw_factor:.2f} ---")    

            dataloaders = {'labeled': trainloader_l, 'unlabeled': trainloader_u}
            criterions = {'ce': criterion_l, 'con': criterion_u}
            writers = {'logger': logger, 'tb': tb}


            if args.local_rank <= 0:        
                if args.co_train:
                    # FIX: Use positional placeholders {} and scientific notation .2e
                    logger.info('===========> Epoch: {:}, backbone1 LR: {:.2e}, backbone2 LR: {:.2e}'\
                                .format(epoch, optimizers['a'].param_groups[0]['lr'], optimizers['b'].param_groups[0]['lr']))                    
                    logger.info('===========> Epoch: {:}, Previous best of ave: {:.2f}, Previous best of branch1: {:.2f}, Previous best of branch2: {:.2f}'\
                                .format(epoch, previous_best, previous_best1, previous_best2))  
                else:
                    # FIX: Also using scientific notation here for consistency
                    logger.info('===========> Epoch: {:}, backbone LR: {:.2e}'\
                                .format(epoch, optimizers.param_groups[0]['lr']))
                

            if args.restore_lr:
                numerator = one_epoch(args, dataloaders, model, optimizers, criterions, ema, epoch, writers, numerator, denominator,schedulers=schedulers,mw_factor=mw_factor)
            else:
                numerator = one_epoch(args, dataloaders, model, optimizers, criterions, ema, epoch, writers,schedulers=schedulers,mw_factor=mw_factor)

        else: 
            if args.ring and epoch%sliding_window_size==0: 
                trainloader_l1, trainloader_l2 = get_subset_dataloaders(args, file_list, epoch, sliding_window_size, length_u)

                if args.restore_lr:
                    print(f'numerator reach : {numerator} and denominator is : {denominator}')
                    numerator=0
                    #restore optimzer lr
                    optimizers['a'].param_groups[0]['lr'] = args.base_lr
                    optimizers['b'].param_groups[0]['lr'] = args.base_lr

            else:
                if args.ring==False:
                    trainloader_l1, trainloader_l2 = trainloader_l, trainloader_l

            mw_duration = 3
            mw_factor=1
            epoch_in_cycle = epoch % sliding_window_size

            if args.ring and args.restore_lr and epoch >= sliding_window_size:
            
                if epoch_in_cycle < mw_duration:
                    # Scale from 10% to 100%
                    mw_factor = 0.1 + (0.9 * (epoch_in_cycle / mw_duration))
                    optimizers['a'].param_groups[0]['lr'] = args.base_lr * mw_factor
                    optimizers['b'].param_groups[0]['lr'] = args.base_lr * mw_factor
                    print(f"--- Micro-Warmup Active: Cycle Epoch {epoch_in_cycle}, Factor: {mw_factor:.2f} ---")    

            dataloaders = {
            'labeled1': trainloader_l1,
            'labeled2': trainloader_l2,
            'unlabeled': trainloader_u
            }
            criterions = {'ce': criterion_l, 'con': criterion_u}
            writers = {'logger': logger, 'tb': tb}

            if args.local_rank <= 0:        
                if args.co_train:
                    # FIX: Use positional placeholders {} and scientific notation .2e
                    logger.info('===========> Epoch: {:}, backbone1 LR: {:.2e}, backbone2 LR: {:.2e}'\
                                .format(epoch, optimizers['a'].param_groups[0]['lr'], optimizers['b'].param_groups[0]['lr']))                    
                    logger.info('===========> Epoch: {:}, Previous best of ave: {:.2f}, Previous best of branch1: {:.2f}, Previous best of branch2: {:.2f}'\
                                .format(epoch, previous_best, previous_best1, previous_best2))  
                else:
                    # FIX: Also using scientific notation here for consistency
                    logger.info('===========> Epoch: {:}, backbone LR: {:.2e}'\
                                .format(epoch, optimizers.param_groups[0]['lr']))
            
            
            if args.restore_lr:            
                numerator = one_epoch_co(args, dataloaders, model, optimizers, criterions, ema, epoch, writers, numerator, denominator,schedulers=schedulers,mw_factor=mw_factor)
            else:
                numerator = one_epoch_co(args, dataloaders, model, optimizers, criterions, ema, epoch, writers,schedulers=schedulers,mw_factor=mw_factor)
        
        print(f"{epoch+1} done.")
        


          # test with different branches
        if args.local_rank <= 0:
            print("Evaluation start")
            if args.ema==True:
                ema.apply_shadow()


            evaluate_result = evaluate_v2(args.local_rank, model, valloader, eval_mode, args)
            if args.co_train:
                accuracy = evaluate_result['accuracy_ave']
                accuracy1 = evaluate_result['accuracy1']
                accuracy2 = evaluate_result['accuracy2']
  
            else:
                accuracy = evaluate_result['accuracy']
 

            



            # --- 추가: val 이 갱신될 때만 두 test 를 잰다 ---
            if accuracy > previous_best:
                r_c = evaluate_v2(args.local_rank, model, testloader_clean, eval_mode, args)
                r_o = evaluate_v2(args.local_rank, model, testloader_obf, eval_mode, args)
                k = 'accuracy_ave' if args.co_train else 'accuracy'
                best_test_clean, best_test_obf = r_c[k], r_o[k]
                best_val_epoch = epoch
                logger.info(
                    f'>>> [BEST val {accuracy:.2f} @ep{epoch}]  '
                    f'test_clean {best_test_clean:.2f}  test_obf {best_test_obf:.2f}  '
                    f'gap {best_test_clean - best_test_obf:+.2f}')
                if tb:
                    tb.add_scalar('test_clean', best_test_clean, epoch)
                    tb.add_scalar('test_obf', best_test_obf, epoch)

            previous_best = save_best_model(args, model, accuracy, previous_best,'best_accuracy_%.2f.pth')


            if args.co_train:
                previous_best1 = save_best_model(args, model, accuracy1, previous_best1, 'best_accuracy1_%.2f.pth')
                previous_best2 = save_best_model(args, model, accuracy2, previous_best2, 'best_accuracy2_%.2f.pth')

            
            if args.ema==True:
                ema.restore()
                

            # Extract classification results
            if args.co_train:
                accuracy = evaluate_result['accuracy_ave']
                accuracy1 = evaluate_result['accuracy1']
                accuracy2 = evaluate_result['accuracy2']
                mean_class_accuracy = evaluate_result['mean_class_accuracy']
                class_accuracy = evaluate_result['class_accuracy']
                
                # --- New Metrics for Co-training ---
                precision = evaluate_result['precision_ave']
                recall = evaluate_result['recall_ave']
                f1 = evaluate_result['f1_ave']


                # Log metrics to TensorBoard
                if tb:
                    tb.add_scalar('accuracy_branch1', accuracy1, epoch)
                    tb.add_scalar('accuracy_branch2', accuracy2, epoch)
                    tb.add_scalar('accuracy_ave', accuracy, epoch)
                    tb.add_scalar('mean_class_accuracy', mean_class_accuracy, epoch)
                    # New TensorBoard logs for precision, recall, and F1-score
                    tb.add_scalar('precision_ave', precision, epoch)
                    tb.add_scalar('recall_ave', recall, epoch)
                    tb.add_scalar('f1_ave', f1, epoch)


                # Log results using logger
                logger.info(f'***** Evaluation with branch 1 {eval_mode} ***** >>>> Accuracy: {accuracy1:.2f}% ')
                logger.info(f'***** Evaluation with branch 2 {eval_mode} ***** >>>> Accuracy: {accuracy2:.2f}% ')
                logger.info(f'***** Evaluation with two branches {eval_mode} ***** >>>> Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%')
                logger.info(f'***** Evaluation {eval_mode} ***** >>>> Mean Class Accuracy: {mean_class_accuracy:.2f}%')
                
                # Log per-class accuracy
                for i, acc in enumerate(class_accuracy):
                    logger.info(f'Class {i} Accuracy: {acc * 100:.2f}%')

            else:
                accuracy = evaluate_result['accuracy']
                mean_class_accuracy = evaluate_result['mean_class_accuracy']
                class_accuracy = evaluate_result['class_accuracy']
                
                # --- New Metrics for Single Branch ---
                precision = evaluate_result['precision']
                recall = evaluate_result['recall']
                f1 = evaluate_result['f1']

                # Log metrics to TensorBoard
                if tb:
                    tb.add_scalar('accuracy', accuracy, epoch)
                    tb.add_scalar('mean_class_accuracy', mean_class_accuracy, epoch)
                    # New TensorBoard logs
                    tb.add_scalar('precision', precision, epoch)
                    tb.add_scalar('recall', recall, epoch)
                    tb.add_scalar('f1', f1, epoch)

                # Log results using logger
                logger.info(f'***** Evaluation {eval_mode} ***** >>>> Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%')
                logger.info(f'***** Evaluation {eval_mode} ***** >>>> Mean Class Accuracy: {mean_class_accuracy:.2f}%')

                # Log per-class accuracy
                for i, acc in enumerate(class_accuracy):
                    logger.info(f'Class {i} Accuracy: {acc * 100:.2f}%')

        

        logger.info('=' * 60)
        logger.info(f'FINAL  p_obf={args.p_obf}  best_val_epoch={best_val_epoch}')
        logger.info(f'  val          {previous_best:.2f}')
        logger.info(f'  test_clean   {best_test_clean:.2f}')
        logger.info(f'  test_obf     {best_test_obf:.2f}')
        logger.info(f'  gap          {best_test_clean - best_test_obf:+.2f}')
        logger.info('=' * 60)


if __name__ == '__main__':
    main(0, args)
