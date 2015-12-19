
local WriteSlice, parent = torch.class('WriteSlice', 'nn.Module')

function WriteSlice:__init(slice)
    parent.__init(self)
    self.slice = slice
    self.gradInput = {}
end

function WriteSlice:updateOutput(input)
    local adjusted_slice = {unpack(self.slice)}
    while #adjusted_slice < input[2]:dim() do
        adjusted_slice = {{}, unpack(adjusted_slice)}
    end
    self.output:resizeAs(input[2]):copy(input[2])
    self.output[adjusted_slice] = input[1]
    return self.output
end

function WriteSlice:updateGradInput(input, gradOutput)
    for i=1,#input do
        self.gradInput[i] = self.gradInput[i] or input[1].new()
        self.gradInput[i]:resizeAs(input[i])
    end
    local adjusted_slice = {unpack(self.slice)}
    while #adjusted_slice < input[2]:dim() do
        adjusted_slice = {{}, unpack(adjusted_slice)}
    end

    self.gradInput[1]:copy(gradOutput[adjusted_slice])
    self.gradInput[2]:copy(gradOutput)
    self.gradInput[2][adjusted_slice]:zero()


    for i=#input+1, #self.gradInput do
        self.gradInput[i] = nil
    end

    return self.gradInput
end
