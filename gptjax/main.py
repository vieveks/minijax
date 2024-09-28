# main.py

import argparse
from train import train_model
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate the GPT model.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode to run: train or evaluate.')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'evaluate':
        evaluate_model()

if __name__ == '__main__':
    main()
