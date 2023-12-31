import argparse

from sbert_finetuning.models import ModelType
from sbert_finetuning.trainer import Trainer
from sbert_finetuning.utils import set_global_seed, init_wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--l_rate", default=1e-7, type=float)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_epoch", default=10, type=int)
    parser.add_argument("--model_type", default=ModelType.SbertLargeNluRu)
    parser.add_argument("--margin", default=10, type=int)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--valid_data_path", required=True)
    parser.add_argument("--test_data_path", required=True)


    parser.add_argument("--wandb", action="store_true")

    return parser.parse_args()


def finetune(l_rate,
             batch_size,
             num_epoch, 
             model_type, 
             device,
             margin,
             export_dir, 
             train_data_path, 
             valid_data_path, 
             test_data_path
             ):
    trainer = Trainer(l_rate, batch_size, num_epoch, model_type, device, margin, export_dir, train_data_path, valid_data_path, test_data_path)

    trainer.train()
    

if __name__ == "__main__":
    set_global_seed(42)
    args = parse_args()

    if args.wandb:
        init_wandb(args)
    

    finetune(
        l_rate=args.l_rate,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        model_type=args.model_type,
        device=args.device,
        margin=args.margin,
        export_dir=args.export_dir,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        test_data_path=args.test_data_path,
    )