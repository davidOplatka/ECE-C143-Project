
import sys
import pathlib

# Ensure the project's `src/` directory is on sys.path so package imports work
# when running this script directly (without installing the package).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

modelName = 'speechBaseline4'

args = {}
args['outputDir'] = '../models/' + modelName
# dataset path should be relative to the project root when running
# `python .\scripts\train_model.py` from the repository root.
args['datasetPath'] = 'data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
#args['speckle_prob'] = 0.3
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)