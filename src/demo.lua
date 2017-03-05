require ('os')
require ('lfs')
require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')

cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = arg[1]
local snapshot = arg[2]
paths.dofile(paths.concat(modelDir, 'config.lua'))
local modelLoadPath = paths.concat(modelDir, snapshot)
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

local testImageDir = arg[3]
local images = {}
for file in lfs.dir(testImageDir) do
    if string.match(file, 'jpg') then
        table.insert(images, file)
    end
end

for idx = 1, #images do
    img = loadAndResizeImage(testImageDir .. images[idx])
    local text, raw = recognizeImageLexiconFree(model, img)
    print(string.format('Recognized text: %s (raw: %s) %s', text, raw, images[idx]))
end
