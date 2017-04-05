function evaluateModel(model, criterion, testSet)
    -- get model parameters

    function validation(input, target, mode)
        --[[ Do validation
        ARGS:
          - `input`  : validation inputs
          - `target` : validation targets
        ]]
        logging(mode)
        model:evaluate()

        -- batch feed forward
        local batchSize = gConfig.valBatchSize
        local nFrame = input:size(1)
        local output = torch.Tensor(nFrame, gConfig.maxT, gConfig.nClasses+1)
        for i = 1, nFrame, batchSize do
            local actualBatchSize = math.min(batchSize, nFrame-i+1)
            local inputBatch = input:narrow(1,i,actualBatchSize)
            local outputBatch = model:forward(inputBatch)
            output:narrow(1,i,actualBatchSize):copy(outputBatch)
        end

        -- compute loss
        local loss = criterion:forward(output, target, true) / nFrame

        -- decoding
        local pred, rawPred = naiveDecoding(output)
        local predStr = label2str(pred)

        -- compute recognition metrics
        local gtStr = label2str(target)
        local nCorrect = 0
        for i = 1, nFrame do
            if predStr[i] == string.lower(gtStr[i]) then
                nCorrect = nCorrect + 1
            end
        end
        local accuracy = nCorrect / nFrame
        logging(string.format('Test loss = %f, accuracy = %f', loss, accuracy))

        -- show prediction examples
        local rawPredStr = label2str(rawPred, true)
        for i = 1, math.min(nFrame, gConfig.nTestDisplay) do
            local idx = math.floor(math.random(1, nFrame))
            logging(string.format('%25s  =>  %-25s  (GT:%-20s)',
                rawPredStr[idx], predStr[idx], gtStr[idx]))
        end
    end

    local testInput, testTarget = testSet:allImageLabel(500)
    validation(testInput, testTarget, 'Testing..')
    collectgarbage()
end
