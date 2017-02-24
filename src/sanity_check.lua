require('os')
require('math')
require('cutorch')
require('optim')
require('utilities')
require('DatasetLmdb')

modelDir = '/opt/crnn/model/m1/'

paths.dofile(paths.concat(modelDir, 'config.lua'))
gConfig = getConfig()
gConfig.modelDir = modelDir
local trainSet = DatasetLmdb("/opt/Synth/val/data.mdb", 10)

print (trainSet:getNumSamples())
local faultyList = {}
local interval = 50000
for i = 1, trainSet:getNumSamples(), interval do
    fl = trainSet:sanitize(i, math.min(trainSet:getNumSamples(), i + interval - 1))
    if #fl ~= 0 then
        for i = 1, #fl do
            faultyList[#faultyList+1] = fl[i]
        end
        print (faultyList)
    end
end
collectgarbage()
