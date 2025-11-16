
modelName = 'speechBaseline4'

args = {}
args['outputDir'] = '../models/' + modelName
args['datasetPath'] = '../data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256 # Number of electrodes * 2
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# --- Time masking hyperparameters ---
args['timeMaskNum'] = 20        # N: number of masks per trial
args['timeMaskMaxFrac'] = 0.075 # M: max mask length as fraction of trial length

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)