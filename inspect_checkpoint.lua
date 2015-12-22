-- simple script that loads a checkpoint and prints its opts

require 'torch'
require 'nn'
require 'nngraph'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a checkpoint and print its options and validation losses.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','model to load')
cmd:option('-gpuid',0,'gpu to use')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 and opt.opencl == 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

if opt.gpuid >= 0 and opt.opencl == 1 then
    print('using OpenCL on GPU ' .. opt.gpuid .. '...')
    require 'cltorch'
    require 'clnn'
    cltorch.setDevice(opt.gpuid + 1)
end

require 'util.ReadSlice'
require 'util.WriteSlice'
local model = torch.load(opt.model)

model_utils = require 'util.model_utils'
params, grad_params = model_utils.combine_all_parameters(model.protos.rnn)

print('opt:')
print(model.opt)
print('val losses:')
print(model.val_losses)

local gradnorm = grad_params:norm()
local paramnorm = params:norm()
local agrad = torch.abs(grad_params)
local aparams = torch.abs(params)
print(string.format("epoch %.3f\ntrain_loss = %.8g\ngrad norm/std/mean/absstd/absmean = %.4g/%.4g/%.4g/%.4g/%.4g\nparam n/s/m/as/am = %.4g/%.4g/%.4g/%.4g/%.4g\ngrad/param norm = %.4g", model.epoch, model.train_losses[model.i],
        gradnorm, grad_params:std(), grad_params:mean(), agrad:std(), agrad:mean(),
        paramnorm, params:std(), params:mean(), aparams:std(), aparams:mean(),
        gradnorm / paramnorm))