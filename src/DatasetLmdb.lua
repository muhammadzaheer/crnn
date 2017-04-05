local pathCache = package.path
package.path = '../third_party/lmdb-lua-ffi/src/?.lua'
local lmdb = require('lmdb')
package.path = pathCache
local Image = require('image')

local DatasetLmdb = torch.class('DatasetLmdb')


function DatasetLmdb:__init(lmdbPath, batchSize, imageType)
    self.batchSize = batchSize or -1
    self.imageType = imageType or 'jpg'
    self:loadDataset(lmdbPath)
end


function DatasetLmdb:loadDataset(lmdbPath)
    self.env = lmdb.environment(lmdbPath, {subdir=false, max_dbs=8, size=1099511627776})
    self.env:transaction(function(txn)
        self.nSamples = tonumber(tostring(txn:get('num-samples')))
    end)
end


function DatasetLmdb:getNumSamples()
    return self.nSamples
end


function DatasetLmdb:getImageGtLexicon(idx, getLexicon)
    getLexicon = getLexicon or false
    local img, label, lexiconList
    self.env:transaction(function(txn)
        local imageKey = string.format('image-%09d', idx)
        local labelKey = string.format('label-%09d', idx)
        local imageBin = tostring(txn:get(imageKey))
        label = tostring(txn:get(labelKey))
        local imageByteLen = string.len(imageBin)
        local imageBytes = torch.ByteTensor(imageByteLen)
        imageBytes:storage():string(imageBin)
        img = Image.decompress(imageBytes, 3, 'byte')
        -- local imgGray = Image.rgb2y(img)
        -- imgGray = Image.scale(imgGray, imgW, imgH)
        -- images[i]:copy(imgGray)
        -- labelList[i] = labelBin
        if getLexicon then
            local lexiconKey = string.format('lexicon-%09d', idx)
            local lexicon = tostring(txn:get(lexiconKey))
            lexiconList = {}
            string.gsub(lexicon, "(%w+)", function (w)
                table.insert(lexiconList, w)
            end)
        end
    end)
    return img, label, lexiconList
end


function DatasetLmdb:allImageLabel(nSampleMax)
    local imgW, imgH = 100, 32
    nSampleMax = nSampleMax or math.huge
    local nSample = math.min(self.nSamples, nSampleMax)
    local images = torch.ByteTensor(nSample, 1, imgH, imgW)
    local labelList = {}
    self.env:transaction(function(txn)
        for i = 1, nSample do
            local imageKey = string.format('image-%09d', i)
            local labelKey = string.format('label-%09d', i)
            local imageBin = tostring(txn:get(imageKey))
            local labelBin = tostring(txn:get(labelKey))
            local imageByteLen = string.len(imageBin)
            local imageBytes = torch.ByteTensor(imageByteLen)
            imageBytes:storage():string(imageBin)
            local img = Image.decompress(imageBytes, 3, 'byte')
            img = Image.rgb2y(img)
            img = Image.scale(img, imgW, imgH)
            images[i]:copy(img)
            labelList[i] = labelBin
        end
    end)
    local labels = str2label(labelList, gConfig.maxT)
    return images, labels
end

function Set (list)
    local set = {}
    for _, l in ipairs(list) do set[l] = true end
    return set
end

-- Hack to ignore faulty jpegs
local faulty = {3610284 ,3610285, 3610286, 3610977, 2529060, 2529063, 5909752,
                1582795, 383637, 383635, 3475418, 3566245, 1291088, 1848340}
local faulty = Set(faulty)
function DatasetLmdb:nextBatch()
    local imgW, imgH = 100, 32
    local randomIndex = torch.LongTensor(self.batchSize):random(1, self.nSamples)
    randomIndex:apply(function(x)
        if faulty[x] then
            x=5
        end
        return x
    end)
    local imageList, labelList = {}, {}
    -- load image binaries and labels
    local success, msg, rc = self.env:transaction(function(txn)
        for i = 1, self.batchSize do
            local idx = randomIndex[i]
            local imageKey = string.format('image-%09d', idx)
            local labelKey = string.format('label-%09d', idx)
            local imageBin = txn:get(imageKey)
            local labelBin = txn:get(labelKey)
            imageList[i] = tostring(imageBin)
            labelList[i] = tostring(labelBin)
        end
    end)

    -- decode images
    local images = torch.ByteTensor(self.batchSize, 1, imgH, imgW)
    for i = 1, self.batchSize do
        local imgBin = imageList[i]
        local imageByteLen = string.len(imgBin)
        local imageBytes = torch.ByteTensor(imageByteLen):fill(0)
        imageBytes:storage():string(imgBin)
        local img = nil
        local status, err = pcall(function() img = Image.decompress(imageBytes, 3, 'byte') end)
        if not status then
            print (err)
            print (randomIndex[i])
            os.exit()
        end
        img = Image.rgb2y(img)
        img = Image.scale(img, imgW, imgH)
        images[i]:copy(img)
    end
    local labels = str2label(labelList, gConfig.maxT)

    collectgarbage()
    return images, labels
end

function DatasetLmdb:sanitize(startIdx, endIdx)
    local imgW, imgH = 100, 32
    local imageList, labelList = {}, {}
    local faultyList = {}
    local curr = 1
    local success, msg, rc = self.env:transaction(function(txn)
        for idx = startIdx, endIdx do
            local imageKey = string.format('image-%09d', idx)
            local labelKey = string.format('label-%09d', idx)
            local imageBin = txn:get(imageKey)
            local labelBin = txn:get(labelKey)
            imageList[curr] = tostring(imageBin)
            labelList[curr] = tostring(labelBin)
            curr = curr + 1
        end
    end)
    -- local images = torch.ByteTensor(endIdx - startIdx + 1, 1, imgH, imgW)
    local faultyCount = 0
    for i = 1, curr-1 do
        local imgBin = imageList[i]
        local imageByteLen = string.len(imgBin)
        local imageBytes = torch.ByteTensor(imageByteLen):fill(0)
        imageBytes:storage():string(imgBin)

        local img = nil
        local status, err = pcall(function () img = Image.decompress(imageBytes, 3, 'byte') end)
        if not status then
            print ('Error at: ', i + startIdx - 1, '(',startIdx, endIdx,')')
            print (err)
            faultyCount = faultyCount + 1
            faultyList[faultyCount] = i + startIdx - 1
        --[[else
            img = Image.rgb2y(img)
            img = Image.scale(img, imgW, imgH)
            images[i]:copy(img)
        --]]
	end
   
        --local img = Image.decompress(imageBytes, 3, 'byte')
        -- img = Image.rgb2y(img)
        -- img = Image.scale(img, imgW, imgH)
        -- images[i]:copy(img)
    end
    print (startIdx, endIdx)
    collectgarbage()
    return faultyList
end
