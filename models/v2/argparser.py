import argparse 

parser = argparse.ArgumentParser()


def parseArguments():
    parser.add_argument('--epochs', type = int, default = 25)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--patience', type = int, default = 5)
    parser.add_argument('--limit', type = int, default = 3)

    parser.add_argument('--train_file', default = '/scratch/scratch2/karthikt/data/train.h5')
    parser.add_argument('--valid_file', default = '/scratch/scratch2/karthikt/data/valid.h5')
    parser.add_argument('--test_file', default = '/scratch/scratch2/karthikt/data/test.h5')

    parser.add_argument('--save_model')
    parser.add_argument('--load_model', default = 'dump')

    parser.add_argument('--finetune', default = 'False')

    parser.add_argument('--save_results')

    parser.add_argument('--plot_folder')
    
    parser.add_argument('--num_maps', type = int)


    return parser.parse_args()
