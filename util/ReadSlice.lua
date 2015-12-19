
local ReadSlice, parent = torch.class('ReadSlice', 'nn.Module')

function ReadSlice:__init(slice)
  parent.__init(self)
  self.slice = slice
end

function ReadSlice:updateOutput(input)
    local adjusted_slice = {unpack(self.slice)}
    while #adjusted_slice < input:dim() do
        adjusted_slice = {{}, unpack(adjusted_slice)}
    end
    local sliced = input[adjusted_slice]
    self.output:resizeAs(sliced):copy(sliced)
    return self.output
end

function ReadSlice:updateGradInput(input, gradOutput)
    if self.gradInput then
        local adjusted_slice = {unpack(self.slice)}
        while #adjusted_slice < input:dim() do
            adjusted_slice = {{}, unpack(adjusted_slice)}
        end
        self.gradInput:resizeAs(input)
        self.gradInput:zero()
        self.gradInput[adjusted_slice]:copy(gradOutput)

        return self.gradInput
    end
end
