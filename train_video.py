import os
import pickle
import numpy as np

import onmt
import onmt.Models
import argparse


parser = argparse.ArgumentParser(description='train_video.py')

parser.add_argument('-layers', type=int, default=2, help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-brnn', action='store_true', help='Use a bidirectional encoder')
parser.add_argument('-rnn_size', type=int, default=500, help='Size of LSTM hidden states')
parser.add_argument('-rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'], help="""The gate type to use in the RNNs""")
parser.add_argument('-encoder_layer', type=str, default='rnn', help="""Type of encoder layer to use. Options: [rnn|mean|transformer]""")
parser.add_argument('-input_size', type=int, default=2000, help='size of video input space')
parser.add_argument('-dropout', type=float, default=0.3, help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-decoder_layer', type=str, default='rnn', help='Type of decoder layer to use. [rnn|transformer]')
parser.add_argument('-coverage_attn', action="store_true", help='Train a coverage attention layer.')
parser.add_argument('-input_feed', type=int, default=1, help="""Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.""")
parser.add_argument('-word_vec_size', type=int, default=2000, help='Word embedding sizes')
parser.add_argument('-context_gate', type=str, default=None, choices=['source', 'target', 'both'], help="""Type of context gate to use [source|target|both]. Do not select for no context gate.""")
parser.add_argument('-attention_type', type=str, default='general', choices=['dot', 'general', 'mlp'], help="""The attention type to use dotprot or general (Luong) or MLP (Bahdanau)""")
parser.add_argument('-copy_attn', action="store_true", help='Train copy attention layer.')
parser.add_argument('-gpus', default=[], nargs='+', type=int, help="Use CUDA on the listed devices.")

opt = parser.parse_args()

print(opt)


def trainModel(model):
    model.train()


    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        mem_loss = onmt.Loss.MemoryEfficientLoss(opt, model.generator,
                                                 criterion,
                                                 copy_loss=opt.copy_attn)

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_stats = onmt.Loss.Statistics()
        report_stats = onmt.Loss.Statistics()

        for i in range(len(trainData)):
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx]
            target_size = batch.tgt.size(0)

            dec_state = None
            trunc_size = opt.truncated_decoder if opt.truncated_decoder \
                else target_size

            for j in range(0, target_size-1, trunc_size):
                trunc_batch = batch.truncate(j, j + trunc_size)

                # Main training loop
                model.zero_grad()
                outputs, attn, dec_state = model(trunc_batch.src,
                                                 trunc_batch.tgt,
                                                 trunc_batch.lengths,
                                                 dec_state)
                batch_stats, inputs, grads \
                    = mem_loss.loss(trunc_batch, outputs, attn)

                torch.autograd.backward(inputs, grads)

                # Update the parameters.
                optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                if dec_state is not None:
                    dec_state.detach()

            report_stats.n_src_words += batch.lengths.data.sum()

            if i % opt.log_interval == -1 % opt.log_interval:
                report_stats.output(epoch, i+1, len(trainData),
                                    total_stats.start_time)
                if opt.log_server:
                    report_stats.log("progress", experiment, optim)
                report_stats = onmt.Loss.Statistics()

        return total_stats


   for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_stats = trainEpoch(epoch)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        #  (2) evaluate on the validation set
        valid_stats = eval(model, criterion, validData)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # Log to remote server.
        if opt.log_server:
            train_stats.log("train", experiment, optim)
            valid_stats.log("valid", experiment, optim)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_stats.ppl(), epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = (model.generator.module.state_dict()
                                if len(opt.gpus) > 1
                                else model.generator.state_dict())
        #  (4) drop a checkpoint
        if epoch >= opt.start_checkpoint_at:
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'optim': optim
            }
            torch.save(checkpoint,
                       '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                       % (opt.save_model, valid_stats.accuracy(),
                          valid_stats.ppl(), epoch))

	

def main():
    encoder = onmt.Models.EncoderVideo(opt)
    decoder = onmt.Models.DecoderVideo(opt)
    model = onmt.Models.NMTModel(encoder, decoder, len(opt.gpus) > 1)
    
    #TODO: what is trainData ?
    trainModel(model, trainData, validData)







if __name__ == "__main__":
    main()
