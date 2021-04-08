from torch.utils.data import DataLoader
import time
import torch
from torch import nn
# import wandb
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from evaluate import evaluate
from data import collate_wrapper
from train_utils import seed_worker
# from torch_geometric.data import DataLoader
import gc
import numpy as np
from torch.nn.functional import one_hot
import random
import numpy
from torch.utils.tensorboard import SummaryWriter
from utils import dump_file, mkdir
from IPython import embed
from data import OurDataset


def train(args, model, optimizer, data):
    train, val, test = data
    #initilize new data model to prevent overwriting
    train_data = OurDataset()
    val_data = OurDataset()
    test_data = OurDataset()

    #split data
    split_frac = 0.7
    index = int(round(len(train.instances)*split_frac))

    # turn on debug to see anomaly like nan
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        train_data.instances = train.instances[:index]
        val_data.instances = val.instances[index:]
        test_data.instances = test.instances[:20]
    else:
        train_data.instances = train.instances[:index]
        val_data.instances = val.instances[index:]
        test_data.instances = test.instances[:20]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper,
                              drop_last=False)

    # model = args.model
    if args.n_gpu > 1 and args.paralell:
        model = nn.DataParallel(model)
    # optimizer = args.optimizer

    # get logger
    logger = args.logger
    writer = args.writer

    train_iterator = range(args.start_epoch, int(args.num_epochs) + args.start_epoch)
    total_steps = int(len(train_loader) * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = None
    if args.scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        # scheduler = CosineAnnealingLR(optimizer, T_max=(int(args.num_epochs) // 4) + 1, eta_min=0)

    logger.debug(f"Total steps: {total_steps}")
    logger.debug(f"Warmup steps: {warmup_steps}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    bad_counter = 0
    best_val_score = -float("inf")
    best_epoch = 0
    t_total = time.time()
    num_steps = 0
    logger.debug(f"{len(train_loader)} steps for each epoch")
    for epoch in train_iterator:
        # logger.debug(f"Epoch {epoch}")
        t = time.time()

        total_loss = 0
        for step, batch in enumerate(train_loader):
            # logger.debug(f"Step {step}")
            num_steps += 1

            batch.in_train = True
            inputs = batch.to(args.device)

            # model learning
            model.train()

            # Mixed Precision (Faster)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = model(inputs, args)
                scaler.scale(loss).backward()

                if (step + 1) % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if args.scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                loss = model(inputs, args)
                loss.backward()
                if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                print("[{:.2f}%] {}/{}, loss: {}".format(100.0*step/len(train_loader), step, len(train_loader), loss.item()), end="\r")

        val_score, output = evaluate(args, model, val_data)

        if epoch > args.burn_in:
            if val_score >= best_val_score:
                best_val_score, best_epoch, bad_counter = val_score, epoch, 0
                torch.save({
                    'epoch': epoch + 1,
                    'num_steps': num_steps + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_score': best_val_score,
                }, args.model_path)
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break

        logger.debug(f'Epoch {epoch} | Train Loss {total_loss:.8f} | Val Score {val_score:.4f} | '
                     f'Time Passed {time.time() - t:.4f}s')
        # embed()

        writer.add_scalar('train', total_loss, epoch)
        writer.add_scalar('val', val_score, epoch)

    logger.debug("Optimization Finished!")
    logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logger.debug('Loading {}th epoch'.format(best_epoch))

    gc.collect()
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    #test_score, output = evaluate(args, model, test_data)

    #logger.debug(f"Test Score {test_score}")

    test_score =0
    # for tensorboard
    #writer.add_scalar('test', test_score, 0)
    writer.add_hparams(
        {'batch_size': args.batch_size, 'num_epochs': args.num_epochs,
         'plm_lr': args.plm_lr, 'lr': args.lr, 'max_grad_norm': args.max_grad_norm, 'dropout': args.dropout,
         # 'model_type': args.model_type,
         },
        {'hparam/test': test_score, 'hparam/val': best_val_score})
    writer.close()
