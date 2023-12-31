import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/" ##TODO 

from pathlib import Path
from argparse import ArgumentParser

import torch
import lightning.pytorch as pl
from data.transforms import CineNetDataTransform
from pl_modules import MriDataModule, CineNetModule #, CRNN_CineNetModule

torch.set_float32_matmul_precision('medium')

def build_args():
    parser = ArgumentParser()
    test_path = "./SingleCoil/Cine/TestSet" #"/input" #TODO
    # test_path = "./SingleCoil/Cine/ValidationSet" #TODO just for inner development, if you need to run reconstruction inference on the validation set
    # test_path = "./SingleCoil/Cine/TrainingSet" #TODO just for inner development, if you need to run reconstruction inference on the validation set

    exp_name = "cinenet-6c" #"Exp1" #TODO
    data_path = Path(test_path)
    default_log_path = Path("inference_logs") / exp_name

    parser.add_argument("--exp_name", default=exp_name, type=str)
    parser.add_argument("--mode", default="test", type=str, choices=["train", "test"]) #TODO default is test here
    parser.add_argument("--model", default="cinenet", type=str, choices=["cinenet", "crnn"]) #TODO default is cinenet here
    parser.add_argument("--ckpt_path", default=None, type=str) #TODO: remember that the argument is --ckpt_path, not --checkpoint_path (fix the README if needed)

    parser = MriDataModule.add_data_specific_args(parser)
    if parser.parse_known_args()[0].model == "cinenet":
        parser = CineNetModule.add_model_specific_args(parser)
    # elif parser.parse_known_args()[0].model == "crnn": #TODO just for now, use cinenet only
    #     parser = CRNN_CineNetModule.add_model_specific_args(parser)
        
    parser.set_defaults(
        data_path=data_path,
        seed=42,
        batch_size=1,
        default_root_dir=default_log_path,
        time_window=12        
    )
    
    args = parser.parse_args()
    
    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
        
    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            verbose=True,
        )
    ]
    
    if args.ckpt_path is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.ckpt_path = ckpt_list[-1]
            
    print(f"Checkpoint path to be used: {args.ckpt_path}")
    
    return args

def main():
    args = build_args()
    pl.seed_everything(args.seed)
    
    #* Data Module
    test_transform = CineNetDataTransform(use_seed=False, time_window=args.time_window)
    
    #* Data Loader
    data_module = MriDataModule(
        data_path=args.data_path,
        test_transform=test_transform,
        test_sample_rate=args.test_sample_rate,
        use_dataset_cache=args.use_dataset_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers, #os.cpu_count()
        distributed_sampler=False
    )
    
    #* Network Model
    if args.model == "cinenet":
        model = CineNetModule(
            num_cascades=args.num_cascades,
            chans=args.chans,
            pools=args.pools,
            dynamic_type=args.dynamic_type,
            weight_sharing=args.weight_sharing,
            data_term=args.data_term,
            lambda_=args.lambda_,
            learnable=args.learnable,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            save_space=args.save_space,
            reset_cache=args.reset_cache,
        )
    # elif args.model == "crnn": #TODO just for now, use cinenet only
    #     model = CRNN_CineNetModule(
    #     num_cascades=args.num_cascades,
    #     chans=args.chans,
    #     pools=args.pools,
    #     dynamic_type=args.dynamic_type,
    #     weight_sharing=args.weight_sharing,
    #     data_term=args.data_term,
    #     lambda_=args.lambda_,
    #     learnable=args.learnable,
    #     lr=args.lr,
    #     lr_step_size=args.lr_step_size,
    #     lr_gamma=args.lr_gamma,
    #     weight_decay=args.weight_decay,
    #     save_space=args.save_space,
    #     reset_cache=args.reset_cache,
    # )
        
    print("Done Loading Data and Model...")

    #* Trainer
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=False,
        callbacks=args.callbacks,
        default_root_dir=args.default_root_dir,
    )
    
    #* Test
    if args.mode == 'test':
        print("Testing "
            f"{(args.model).upper()} with "
            f"{args.num_cascades} unrolled iterations.\n")
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    
if __name__ == '__main__':
    main()