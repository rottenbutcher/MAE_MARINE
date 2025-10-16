import torch
import torch.nn as nn
import os
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    start_epoch = 0
    if args.resume:
        start_epoch, _ = builder.resume_model(base_model, args, logger = logger)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            
            # =================== 입력 데이터 형태에 따른 분기 처리 ===================
            if isinstance(data, tuple):
                # HPR 모드 (MAE, M2AE): partial_view, ground_truth 2개 입력
                partial_view, ground_truth = data
                partial_view = partial_view.cuda()
                ground_truth = ground_truth.cuda()
                loss = base_model(partial_view, ground_truth)
            else:
                # Autoencoder 모드 (dVAE): points 1개 입력
                points = data.cuda()
                loss = base_model(points)
            # =================================================================

            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()

            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
            losses.update([loss.item()*1000])

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log(f"[Epoch {epoch}/{config.max_epoch}][Batch {idx+1}/{n_batches}] "
                          f"BatchTime = {batch_time.val():.3f}s DataTime = {data_time.val():.3f}s "
                          f"Loss = {losses.val(0):.4f} lr = {optimizer.param_groups[0]['lr']:.6f}", logger=logger)
        
        if isinstance(scheduler, list):
            for item in scheduler: item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
        print_log(f"[Training] EPOCH: {epoch} EpochTime = {epoch_end_time - epoch_start_time:.3f}s "
                  f"Loss = {losses.avg(0):.4f} lr = {optimizer.param_groups[0]['lr']:.6f}", logger=logger)

        if epoch % 25 == 0 or epoch == config.max_epoch:
             builder.save_checkpoint(base_model, optimizer, epoch, None, None, f'ckpt-epoch-{epoch:03d}', args, logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger = logger)
        
    if train_writer is not None: train_writer.close()
    if val_writer is not None: val_writer.close()