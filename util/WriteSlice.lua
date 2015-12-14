
local WriteSlice, parent = torch.class('WriteSlice', 'nn.Module')

function WriteSlice:__init(slice)
  parent.__init(self)
  self.slice = slice
end

function WriteSlice:updateOutput(input)
    self.output:resizeAs(input[2]):copy(input[2])
    self.output[self.slice] = input[1] 
    return self.output
end

function WriteSlice:updateGradInput(input, gradOutput)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:copy(gradOutput[self.slice])

    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:copy(gradOutput)
    self.gradInput[2][self.slice]:zero()


    for i=#input+1, #self.gradInput do
        self.gradInput[i] = nil
    end

    return self.gradInput
end
