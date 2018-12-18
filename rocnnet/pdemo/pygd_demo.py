import sys
import time
import argparse

import numpy as np

import llo.llo as llo
import rocnnet.rocnnet as rcn

prog_description = 'Demo gd_trainer'

def batch_generate(n, batchsize):
    total = n * batchsize
    return np.random.rand(total)

def avgevry2(indata):
    return (indata[0::2] + indata[1::2]) / 2

def str2bool(opt):
    optstr = opt.lower()
    if optstr in ('yes', 'true', 't', 'y', '1'):
        return True
    elif optstr in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    default_ts = time.time()

    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--seed', dest='seed',
        type=str2bool, nargs='?', const=False, default=True,
        help='Whether to seed or not (default: True)')
    parser.add_argument('--seedval', dest='seedval', type=int, nargs='?', default=int(default_ts),
        help='Random seed value (default: <current time>)')
    parser.add_argument('--n_train', dest='n_train', type=int, nargs='?', default=3000,
        help='Number of times to train (default: 3000)')
    parser.add_argument('--n_test', dest='n_test', type=int, nargs='?', default=500,
        help='Number of times to test (default: 500)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='rocnnet/pretrained/gdmodel.pbx',
        help='Filename to load pretrained model (default: rocnnet/pretrained/gdmodel.pbx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        llo.seed(args.seedval)
        np.random.seed(args.seedval)

    n_in = 10
    n_out = n_in / 2

    hiddens = [
        rcn.get_layer(rcn.sigmoid, 9),
        rcn.get_layer(rcn.sigmoid, n_out)
    ]

    brain = rcn.get_mlp(n_in, hiddens, "brain")
    untrained_brain = brain.copy()
    try:
        with open(args.load, 'rb') as f:
            pretrained_brain = brain.parse_from_string(f.read())
    except:
        pretrained_brain = brain.copy()

    n_batch = 3
    show_every_n = 500
    trainer = rcn.GDTrainer(brain, rcn.get_sgd(0.9), n_batch, "gdn")

    start = time.time()
    for i in range(args.n_train):
        if i % show_every_n == show_every_n - 1:
            trained_derr = llo.evaluate(trainer.error())
            print('training {}\ntraining error:\n{}'
                .format(i + 1, trained_derr))
        batch = batch_generate(n_in, n_batch)
        batch_out = avgevry2(batch)
        trainer.train(batch, batch_out)
    end = time.time()
    print('training time: {} seconds'.format(end - start))

    # exit code:
    #	0 = fine
    #	1 = training error rate is wrong
    untrained_err = 0
    trained_err = 0
    pretrained_err = 0

    testin = llo.variable(np.zeros([n_in], dtype=float), "testin")
    untrained_out = untrained_brain.forward(testin)
    trained_out = brain.forward(testin)
    pretrained_out = pretrained_brain.forward(testin)

    for i in range(args.n_test):
        if i % show_every_n == show_every_n - 1:
            print('testing {}'.format(i + 1))

        test_batch = batch_generate(n_in, 1)
        test_batch_out = avgevry2(test_batch)
        testin.assign(test_batch)

        untrained_data = llo.evaluate(untrained_out)
        trained_data = llo.evaluate(trained_out)
        pretrained_data = llo.evaluate(pretrained_out)

        untrained_err += np.mean(abs(untrained_data - test_batch_out))
        trained_err += np.mean(abs(trained_data - test_batch_out))
        pretrained_err += np.mean(abs(pretrained_data - test_batch_out))

    untrained_err /= args.n_test
    trained_err /= args.n_test
    pretrained_err /= args.n_test
    print('untrained mlp error rate: {}%'.format(untrained_err * 100))
    print('trained mlp error rate: {}%'.format(trained_err * 100))
    print('pretrained mlp error rate: {}%'.format(pretrained_err * 100))

    try:
        with open(args.save, 'wb') as f:
            f.write(brain.serialize_to_string(trained_out))
    except:
        pass

if '__main__' == __name__:
    main(sys.argv[1:])
