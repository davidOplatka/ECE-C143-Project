# Model configuration for TDS Conv-LSTM w/ reduced threshold crossings
modelName = 'conv-lstm-reduced-tcs'

args = {}
args['outputDir'] = '../models/' + modelName
args['datasetPath'] = '../data/ptDecoder_ctc'
args['rnn_type'] = 'lstm'
args['use_tds'] = True
args['num_tds_blocks'] = 3
args['tds_channels'] = 64
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['optimizer'] = 'ADAM'
args['warmupSteps'] = 500
args['decayType'] = 'linear'
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['optimizerEps'] = 0.1
args['SGDMomentum'] = 0
args['l2_decay'] = 1e-5
args['nUnits'] = 256
args['nBatch'] = 15000
args['patience'] = 500
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 242
args['nThresholdCrossings'] = 114
args['nSpikeBandPowers'] = 128
args['speckle_prob'] = 0
args['max_grad_norm'] = 5.0
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['timeMaskNum'] = 4
args['timeMaskMaxFrac'] = 0.08

from neural_decoder.neural_decoder_trainer import trainModel
trainModel(args)

"""
Model configuration for Baseline
"""
# modelName = 'speechBaseline4'

# args = {}
# args['outputDir'] = '../models/' + modelName
# args['datasetPath'] = '../data/ptDecoder_ctc'
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 128
# args['optimizer'] = 'ADAM'
# args['warmupSteps'] = 0
# args['decayType'] = 'linear'
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['optimizerEps'] = 0.1
# args['SGDMomentum'] = 0
# args['l2_decay'] = 1e-5
# args['nUnits'] = 256
# args['nBatch'] = 15000
# args['patience'] = 500
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['nThresholdCrossings'] = 128
# args['nSpikeBandPowers'] = 128
# args['dropout'] = 0.2
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False

# from neural_decoder.neural_decoder_trainer import trainModel
# trainModel(args)


"""
Model configuration for GRU
"""
# modelName = 'gru'

# args = {}
# args['outputDir'] = '../models/' + modelName
# args['datasetPath'] = '../data/ptDecoder_ctc'
# args['rnn_type'] = 'gru'
# args['use_tds'] = False
# args['num_tds_blocks'] = 0
# args['tds_channels'] = 0
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 128
# args['optimizer'] = 'ADAM'
# args['warmupSteps'] = 500
# args['decayType'] = 'linear'
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['optimizerEps'] = 0.1
# args['SGDMomentum'] = 0
# args['l2_decay'] = 1e-5
# args['nUnits'] = 256
# args['nBatch'] = 15000
# args['patience'] = 500
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['nThresholdCrossings'] = 128
# args['nSpikeBandPowers'] = 128
# args['speckle_prob'] = 0
# args['max_grad_norm'] = 5.0
# args['dropout'] = 0.2
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['timeMaskNum'] = 4
# args['timeMaskMaxFrac'] = 0.08

# from neural_decoder.neural_decoder_trainer import trainModel
# trainModel(args)

"""
Model configuration for LSTM
"""
# modelName = 'lstm'

# args = {}
# args['outputDir'] = '../models/' + modelName
# args['datasetPath'] = '../data/ptDecoder_ctc'
# args['rnn_type'] = 'gru'
# args['use_tds'] = False
# args['num_tds_blocks'] = 0
# args['tds_channels'] = 0
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 128
# args['optimizer'] = 'ADAM'
# args['warmupSteps'] = 500
# args['decayType'] = 'linear'
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['optimizerEps'] = 0.1
# args['SGDMomentum'] = 0
# args['l2_decay'] = 1e-5
# args['nUnits'] = 256
# args['nBatch'] = 15000
# args['patience'] = 500
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['nThresholdCrossings'] = 128
# args['nSpikeBandPowers'] = 128
# args['speckle_prob'] = 0
# args['max_grad_norm'] = 5.0
# args['dropout'] = 0.2
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['timeMaskNum'] = 4
# args['timeMaskMaxFrac'] = 0.08

# from neural_decoder.neural_decoder_trainer import trainModel
# trainModel(args)

"""
Model configuration for TDS Conv-GRU
"""
# modelName = 'conv-gru'

# args = {}
# args['outputDir'] = '../models/' + modelName
# args['datasetPath'] = '../data/ptDecoder_ctc'
# args['rnn_type'] = 'gru'
# args['use_tds'] = True
# args['num_tds_blocks'] = 3
# args['tds_channels'] = 64
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 128
# args['optimizer'] = 'ADAM'
# args['warmupSteps'] = 500
# args['decayType'] = 'linear'
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['optimizerEps'] = 0.1
# args['SGDMomentum'] = 0
# args['l2_decay'] = 1e-5
# args['nUnits'] = 256
# args['nBatch'] = 15000
# args['patience'] = 500
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['nThresholdCrossings'] = 128
# args['nSpikeBandPowers'] = 128
# args['speckle_prob'] = 0
# args['max_grad_norm'] = 5.0
# args['dropout'] = 0.2
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['timeMaskNum'] = 4
# args['timeMaskMaxFrac'] = 0.08

# from neural_decoder.neural_decoder_trainer import trainModel
# trainModel(args)

"""
Model configuration for TDS Conv-LSTM
"""
# modelName = 'conv-lstm'

# args = {}
# args['outputDir'] = '../models/' + modelName
# args['datasetPath'] = '../data/ptDecoder_ctc'
# args['rnn_type'] = 'lstm'
# args['use_tds'] = True
# args['num_tds_blocks'] = 3
# args['tds_channels'] = 64
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 128
# args['optimizer'] = 'ADAM'
# args['warmupSteps'] = 500
# args['decayType'] = 'linear'
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['optimizerEps'] = 0.1
# args['SGDMomentum'] = 0
# args['l2_decay'] = 1e-5
# args['nUnits'] = 256
# args['nBatch'] = 15000
# args['patience'] = 500
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['nThresholdCrossings'] = 128
# args['nSpikeBandPowers'] = 128
# args['speckle_prob'] = 0
# args['max_grad_norm'] = 5.0
# args['dropout'] = 0.2
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['timeMaskNum'] = 4
# args['timeMaskMaxFrac'] = 0.08

# from neural_decoder.neural_decoder_trainer import trainModel
# trainModel(args)

"""
Model configuration for TDS Conv-LSTM w/ reduced spike band powers
"""
# modelName = 'conv-lstm-reduced-spb'

# args = {}
# args['outputDir'] = '../models/' + modelName
# args['datasetPath'] = '../data/ptDecoder_ctc'
# args['rnn_type'] = 'lstm'
# args['use_tds'] = True
# args['num_tds_blocks'] = 3
# args['tds_channels'] = 64
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 128
# args['optimizer'] = 'ADAM'
# args['warmupSteps'] = 500
# args['decayType'] = 'linear'
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['optimizerEps'] = 0.1
# args['SGDMomentum'] = 0
# args['l2_decay'] = 1e-5
# args['nUnits'] = 256
# args['nBatch'] = 15000
# args['patience'] = 500
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 242
# args['nThresholdCrossings'] = 128
# args['nSpikeBandPowers'] = 114
# args['speckle_prob'] = 0
# args['max_grad_norm'] = 5.0
# args['dropout'] = 0.2
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['timeMaskNum'] = 4
# args['timeMaskMaxFrac'] = 0.08

# from neural_decoder.neural_decoder_trainer import trainModel
# trainModel(args)


"""
Model configuration for TDS Conv-LSTM w/ reduced spike band powers and threshold crossings
"""
# modelName = 'conv-lstm-reduced-features'

# args = {}
# args['outputDir'] = '../models/' + modelName
# args['datasetPath'] = '../data/ptDecoder_ctc'
# args['rnn_type'] = 'lstm'
# args['use_tds'] = True
# args['num_tds_blocks'] = 3
# args['tds_channels'] = 64
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 128
# args['optimizer'] = 'ADAM'
# args['warmupSteps'] = 500
# args['decayType'] = 'linear'
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['optimizerEps'] = 0.1
# args['SGDMomentum'] = 0
# args['l2_decay'] = 1e-5
# args['nUnits'] = 256
# args['nBatch'] = 15000
# args['patience'] = 500
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 228
# args['nThresholdCrossings'] = 114
# args['nSpikeBandPowers'] = 114
# args['speckle_prob'] = 0
# args['max_grad_norm'] = 5.0
# args['dropout'] = 0.2
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['timeMaskNum'] = 4
# args['timeMaskMaxFrac'] = 0.08

# from neural_decoder.neural_decoder_trainer import trainModel
# trainModel(args)
