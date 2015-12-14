
local ReadSlice, parent = torch.class('ReadSlice', 'nn.Module')

function ReadSlice:__init(slice)
  parent.__init(self)
  self.slice = slice
end

function ReadSlice:updateOutput(input)
    local sliced = input[self.slice]
    self.output:resizeAs(sliced):copy(sliced)
    return self.output
end

function ReadSlice:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput or input.new()
    self.gradInput:zero()[self.slice]:copy(gradOutput)

    return self.gradInput
end
